import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy
from timm.scheduler import create_scheduler
from models import build_classification_model, save_checkpoint, build_segmentation_model
import utils
from utils import metric_AUROC

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch_classification,train_one_epoch_segmentation, test_classification,evaluate_classification,evaluate_segmentation,test_segmentation
from utils import learning_rate_schedule,dice,mean_dice_coef

try:
  import habana_frameworks.torch.core as htcore
except:
  pass
  
sys.setrecursionlimit(40000)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    ds = torch.utils.data.get_worker_info().dataset
    tf = getattr(ds, "train_transform", None)
    if tf is not None and hasattr(tf, "set_random_seed"):
        tf.set_random_seed(seed)

def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  utils.init_distributed_mode(args)
  device = torch.device(args.device)
  
  if args.device == 'cuda':
    cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  # training phase
  if args.mode == "train":
    if args.distributed:
      train_sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
      val_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)

      data_loader_train = DataLoader(dataset=dataset_train, sampler=train_sampler, batch_size=args.batch_size,
                                   num_workers=args.workers, pin_memory=True)
      data_loader_val = DataLoader(dataset=dataset_val, sampler=val_sampler, batch_size=args.batch_size,
                                  num_workers=args.workers, pin_memory=True)
    else:
      data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
      data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    log_file = os.path.join(model_path, "models.log")

    # training phase
    print("start training....")
    for i in range(args.start_index, args.num_trial):
      print("run:",str(i+1))
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      if "convnext" in args.model_name.lower() or "swin" in args.model_name.lower() or "vit" in args.model_name.lower():
        criterion = torch.nn.BCEWithLogitsLoss()
      else:
        criterion = torch.nn.BCELoss()
      model = build_classification_model(args)
      print(model)
      model.to(device)
      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      if args.distributed:
        if args.device == 'cuda':
          model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
          model = nn.parallel.DistributedDataParallel(model)
      
      if args.torch_compile:
        model = torch.compile(model, backend="hpu_backend")
      
      if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
      elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
      elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=False)
      if args.lr_Scheduler.lower() == "reducelronplateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
                                         threshold=0.0001, min_lr=0)
      elif args.lr_Scheduler.lower() == "cosine":
        args.sched = "cosine"
        lr_scheduler, _ = create_scheduler(args, optimizer)
      if args.resume:
        resume = os.path.join(model_path, experiment + '.pth.tar')
        if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume, weights_only=False)

          start_epoch = checkpoint['epoch']
          init_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, init_loss))
        else:
          print("=> no checkpoint found at '{}'".format(args.resume))

      for epoch in range(start_epoch, args.num_epoch):
        if args.distributed:
          data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch_classification(data_loader_train, device, model, criterion, optimizer, epoch, args, apply_activation=False)
        val_loss = evaluate_classification(data_loader_val, device, model, criterion, args, apply_activation=False)
        print("Update Learning Rate")
        lr_scheduler.step(val_loss)

        if args.distributed:
          cpu_model_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
        else:
          cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
          
        cpu_optimizer_state = {}
        
        for key, value in optimizer.state_dict().items():
          if key == 'state':
              # Optimizer tracking states (momentum, etc.) are tensors
              cpu_optimizer_state['state'] = {
                  param_id: {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in param_states.items()}
                  for param_id, param_states in value.items()
              }
          else:
              # Hyperparameters like learning rate are standard Python objects
              cpu_optimizer_state[key] = value
        
        if val_loss < best_val_loss:
          save_checkpoint({
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'state_dict': cpu_model_state,
            'optimizer': cpu_optimizer_state,
            'scheduler': lr_scheduler.state_dict(),
          },  filename=save_model_path)
          patience_counter = 0
          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                               save_model_path))
          best_val_loss = val_loss

        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss ))
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break

      # log experiment
      with open(log_file, 'a') as f:
        f.write(experiment + "\n")
        f.close()

  if args.distributed:
    utils.cleanup()

  if utils.is_main_process():
    print("start testing.....")
    output_file = os.path.join(output_path, args.exp_name + "_results.txt")
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    log_file = os.path.join(model_path, "models.log")
    if not os.path.isfile(log_file):
      print("log_file ({}) not exists!".format(log_file))
    else:
      mean_auc = []
      with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
        experiment = reader.readline()
        print(">> Disease = {}".format(diseases))
        writer.write("Disease = {}\n".format(diseases))
        while experiment:
          experiment = experiment.replace('\n', '')
          saved_model = os.path.join(model_path, experiment + ".pth.tar")
          y_test, p_test = test_classification(saved_model, data_loader_test, device, args)
          if test_diseases is not None:
            y_test = copy.deepcopy(y_test[:,test_diseases])
            p_test = copy.deepcopy(p_test[:, test_diseases])
            individual_results = metric_AUROC(y_test, p_test, len(test_diseases))
          else:
            individual_results = metric_AUROC(y_test, p_test, args.num_class)
          print(">>{}: AUC = {}".format(experiment, np.array2string(np.array(individual_results), precision=4, separator=',')))
          writer.write(
            "{}: AUC = {}\n".format(experiment, np.array2string(np.array(individual_results), precision=4, separator='\t')))
          mean_over_all_classes = np.array(individual_results).mean()
          print(">>{}: AUC = {:.4f}".format(experiment, mean_over_all_classes))
          writer.write("{}: AUC = {:.4f}\n".format(experiment, mean_over_all_classes))
          mean_auc.append(mean_over_all_classes)
          experiment = reader.readline()
        mean_auc = np.array(mean_auc)
        print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
        writer.write("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
        print(">> Mean AUC over All trials: = {:.4f}".format(np.mean(mean_auc)))
        writer.write("Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)))
        print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
        writer.write("STD over All trials:  = {:.4f}\n".format(np.std(mean_auc)))

def segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion):
  utils.init_distributed_mode(args)
  device = torch.device(args.device)

  if not os.path.exists(model_path):
    os.makedirs(model_path)
  logs_path = os.path.join(model_path, "Logs")
  if not os.path.exists(logs_path):
    os.makedirs(logs_path)
  if os.path.exists(os.path.join(logs_path, "log.txt")):
    log_writer = open(os.path.join(logs_path, "log.txt"), 'w')
  else:
    log_writer = open(os.path.join(logs_path, "log.txt"), 'w')

  model, optimizer, loss_scaler = build_segmentation_model(args, device, log_writer)

  if args.mode == "train":
    start_num_epochs = 0

    if args.distributed:
      train_sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
      val_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)

      data_loader_train = DataLoader(dataset=dataset_train, sampler=train_sampler, batch_size=args.train_batch_size,
                                   num_workers=args.train_num_workers, drop_last=False, pin_memory=True)
      data_loader_val = DataLoader(dataset=dataset_val, sampler=val_sampler, batch_size=args.train_batch_size,
                                  num_workers=args.train_num_workers, drop_last=False, pin_memory=True)
    else:            
      data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                  num_workers=args.train_num_workers, drop_last=False, pin_memory=True)
      data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.train_batch_size,
                                                        shuffle=False, num_workers=args.train_num_workers,drop_last=False, pin_memory=True)
    best_val_loss = 100000
    patience_counter = 0
    for epoch in range(start_num_epochs, args.epochs):
        lr_ = learning_rate_schedule(epoch, args.epochs, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'], epoch), file=log_writer)
        loss_avg = train_one_epoch_segmentation(model, data_loader_train, device, optimizer, loss_scaler, epoch, criterion, log_writer, args)
        print('Training loss: {}@Epoch: {}'.format(loss_avg, epoch), file=log_writer)
        log_writer.flush()
        val_avg = evaluate_segmentation(model, data_loader_val, device, epoch, criterion, log_writer, args)
        
        if val_avg < best_val_loss:
          if args.distributed:
            cpu_model_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
          else:
            cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
          
          cpu_optimizer_state = {}
          
          for key, value in optimizer.state_dict().items():
            if key == 'state':
                # Optimizer tracking states (momentum, etc.) are tensors
                cpu_optimizer_state['state'] = {
                    param_id: {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in param_states.items()}
                    for param_id, param_states in value.items()
                }
            else:
                # Hyperparameters like learning rate are standard Python objects
                cpu_optimizer_state[key] = value
            
          torch.save({
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'state_dict': cpu_model_state,
            'optimizer': cpu_optimizer_state,
          }, os.path.join(model_path, "checkpoint.pt"))

          print( "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss,val_avg, os.path.join(model_path, "checkpoint.pt")), file=log_writer)
          best_val_loss = val_avg
          patience_counter = 0
        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss), file=log_writer)
          patience_counter += 1
        if patience_counter > args.patience:
          print("Early Stopping", file=log_writer)
          break

        log_writer.flush()

  if args.distributed:
    utils.cleanup()
  
  if args.device == 'cuda':
    torch.cuda.empty_cache()
  
  if utils.is_main_process():
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.test_num_workers,drop_last=False, pin_memory=True)
    test_y, test_p = test_segmentation(args, device, os.path.join(model_path, "checkpoint.pt"), data_loader_test, log_writer)

    print("[INFO] Dice = {:.2f}%".format(100.0 * dice(test_p, test_y)), file=log_writer)
    print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writer)
    log_writer.flush()


