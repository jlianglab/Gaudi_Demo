import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy


from models import build_omni_model, save_checkpoint
from utils import metric_AUROC, cosine_scheduler, plot, init_distributed_mode, is_main_process
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch, test_classification, evaluate, test_gather
#import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from functools import partial
import torch.nn as nn
import json

try:
    import habana_frameworks.torch.core as htcore
except:
    pass

sys.setrecursionlimit(40000)

def omni_engine(args, model_path, output_path, dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list):
    init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # logs
    exp = 'Ark_Plus'
    for dataset in dataset_list:
        exp += '_' + dataset 
    model_path = os.path.join(model_path, exp)
    model_path = os.path.join(model_path, args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(model_path, "train.log")
    output_file = os.path.join(output_path, exp+"_"+args.exp_name+"_results.txt")

    # dataloaders for pretraining
    data_loader_list_train = []
    for d in dataset_train_list:
        if args.distributed:
            train_sampler = torch.utils.data.DistributedSampler(d, shuffle=True)
            data_loader_list_train.append(DataLoader(dataset=d, sampler=train_sampler, batch_size=args.batch_size,
                                            num_workers=args.workers, pin_memory=True, drop_last=True))
        else:
            data_loader_list_train.append(DataLoader(dataset=d, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers, pin_memory=True))
    data_loader_list_val = []
    for dv in dataset_val_list:
        if args.distributed:
            val_sampler = torch.utils.data.DistributedSampler(dv, shuffle=False)
            data_loader_list_val.append(DataLoader(dataset=dv, sampler=val_sampler, batch_size=args.batch_size,
                                            num_workers=args.workers, pin_memory=True))
        else:
            data_loader_list_val.append(DataLoader(dataset=dv, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers, pin_memory=True))
    data_loader_list_test = []
    dataset_test_len = []
    for dt in dataset_test_list:
        if args.distributed:
            dataset_test_len.append(len(dt))
            test_sampler = torch.utils.data.DistributedSampler(dt, shuffle=False) 
            data_loader_list_test.append(DataLoader(dataset=dt, sampler=test_sampler, batch_size=int(args.batch_size/2),
                                            num_workers=args.workers, pin_memory=True))
        else:
            data_loader_list_test.append(DataLoader(dataset=dt, batch_size=int(args.batch_size/2), shuffle=False,
                                        num_workers=args.workers, pin_memory=True))

    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in dataset_list]
    print("num_classes_list:", num_classes_list)


    # training setups
    model = build_omni_model(args, num_classes_list)
    teacher = build_omni_model(args, num_classes_list)     
    print(model)

    model.to(device)
    teacher.to(device)
    
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model_name} network.")
    
    if args.distributed:
        #avoid error from unused heads
        if args.device == 'cuda':
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    if args.torch_compile:
        model = torch.compile(model,backend="hpu_backend")
        teacher = torch.compile(teacher,backend="hpu_backend")
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    if args.ema_mode == "epoch":
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, len(dataset_list))
    elif args.ema_mode == "iteration":
        iters_per_epoch = 0
        for d in data_loader_list_train:
            iters_per_epoch += len(d)
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.pretrain_epochs, iters_per_epoch)
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    save_model_path = os.path.join(model_path, exp)

    if args.mode == "train":
        auc_dict = {}
        
        for dataset in dataset_list:
            auc_dict[dataset] = {'student':[], 'teacher':[], 'student_best_auc':0, 'teacher_best_auc':0}
        
        if args.resume:
            resume = save_model_path + '.pth.tar'
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
                start_epoch = checkpoint['epoch']
                init_loss = checkpoint['lossMIN']
                state_dict = checkpoint['state_dict']
                teacher_state_dict = checkpoint['teacher']
                
                if not args.distributed:
                    for k in list(state_dict.keys()):
                        if k.startswith('module.'):
                          state_dict[k[len("module."):]] = state_dict[k]
                          del state_dict[k]
                    
                    for k in list(teacher_state_dict.keys()):
                        if k.startswith('module.'):
                          teacher_state_dict[k[len("module."):]] = teacher_state_dict[k]
                          del teacher_state_dict[k]

                model.load_state_dict(state_dict, strict=True)
                teacher.load_state_dict(teacher_state_dict, strict=True)
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                        .format(resume, start_epoch, init_loss))
                start_epoch += 1
                
                for dataset in dataset_list:
                    with open(os.path.join(output_path, f'{dataset}_auc.json'), 'r') as f:
                        auc_dict[dataset] = json.load(f)
                
            else:
                print("=> no checkpoint found at '{}'".format(resume))

        with open(log_file, 'a') as log:
            log.write(str(args))
        log.close()

        test_results,test_results_teacher = [],[]
            
        it = start_epoch * len(dataset_list)
        
        for epoch in range(start_epoch, args.pretrain_epochs):
            for i, data_loader in enumerate(data_loader_list_train):
                if args.distributed:
                    data_loader.sampler.set_epoch(epoch)
                criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                train_one_epoch(model, i, dataset_list[i], data_loader, device, criterion, optimizer, epoch, args.ema_mode, teacher, momentum_schedule, it, args.lazy_mode)
                it += 1
                
                with open(output_file, 'a') as writer:
                    if is_main_process():
                        writer.write("Omni-pretraining stage:\n")
                        writer.write("Epoch {:04d}:\n".format(epoch))
                    t_res, t_res_teacher = [],[]
                    for i, dataset in enumerate(dataset_list):
                        diseases = datasets_config[dataset]['diseases']
                        multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                        y_test, p_test, idx_test = test_classification(model, i, data_loader_list_test[i], device, args.lazy_mode, multiclass)
                        y_test_teacher, p_test_teacher, idx_test_teacher = test_classification(teacher, i, data_loader_list_test[i], device, args.lazy_mode, multiclass)
                        
                        if args.distributed:
                            y_test, p_test = test_gather(y_test, p_test, idx_test, dataset_test_len[i], device)
                            y_test_teacher, p_test_teacher = test_gather(y_test_teacher, p_test_teacher, idx_test_teacher, dataset_test_len[i], device)
                        
                        if is_main_process():
                            print(">>{} Disease = {}".format(dataset, diseases))
                            writer.write("{} Disease = {}\n".format(dataset, diseases))

                            if multiclass:
                                acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                                acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                                print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset,acc, acc_teacher))
                                writer.write(
                                    "\n{}: Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset, np.array2string(np.array(acc), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))   
                                t_res.append(acc)
                                t_res_teacher.append(acc_teacher)

                            if dataset == "CheXpert":
                                test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                                test_diseases = [diseases.index(c) for c in test_diseases_name]
                                y_test = copy.deepcopy(y_test[:,test_diseases])
                                p_test = copy.deepcopy(p_test[:, test_diseases])
                                individual_results = metric_AUROC(y_test, p_test, len(test_diseases)) 
                                y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                                p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases)) 
                            else: 
                                individual_results = metric_AUROC(y_test, p_test, len(diseases))
                                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases)) 
                            
                            print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            writer.write(
                                "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            mean_over_all_classes = np.array(individual_results).mean()
                            mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                            print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                            writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                            t_res.append(mean_over_all_classes)
                            t_res_teacher.append(mean_over_all_classes_teacher)
                            
                            auc_dict[dataset]['student'].append(mean_over_all_classes)
                            auc_dict[dataset]['teacher'].append(mean_over_all_classes_teacher)
                            
                            if mean_over_all_classes > auc_dict[dataset]['student_best_auc']:
                                auc_dict[dataset]['student_best_auc'] = mean_over_all_classes
                            if mean_over_all_classes_teacher > auc_dict[dataset]['teacher_best_auc']:
                                auc_dict[dataset]['teacher_best_auc'] = mean_over_all_classes_teacher
                        
                    writer.close()

                    if is_main_process():
                        test_results.append(t_res)
                        test_results_teacher.append(t_res_teacher)
                        
                        print("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(test_results, test_results_teacher))
                
            val_loss_list = []
            for i, dv in enumerate(data_loader_list_val):
                criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                val_loss = evaluate(model, i, dv, device, criterion, dataset_list[i], args.lazy_mode, args.distributed)
                val_loss_list.append(val_loss)
            
            avg_val_loss = np.average(val_loss_list)
            if args.val_loss_metric == "average":
                val_loss_metric = avg_val_loss
            else:
                val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
            lr_scheduler.step(val_loss_metric)
            
            cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            cpu_teacher_state = {k: v.cpu() for k, v in teacher.state_dict().items()}
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
            
            print("Epoch {:04d}: avg_val_loss {:.5f}, saving model to {}".format(epoch, avg_val_loss,save_model_path))

            if is_main_process():
                save_checkpoint({
                        'epoch': epoch,
                        'lossMIN': val_loss_list,
                        'state_dict': cpu_model_state,
                        'teacher': cpu_teacher_state,
                        'optimizer': cpu_optimizer_state,
                        'scheduler': lr_scheduler.state_dict(),
                        },  filename=save_model_path)

                with open(log_file, 'a') as log:
                    log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
                    log.write("     Datasets  : " + str(dataset_list) + "\n")
                    log.write("     Val Losses: " + str(val_loss_list) + "\n")
                    log.close()
                
                cycles = range(1, epoch+2)
                plot(auc_dict, cycles, dataset_list, output_path)
                
                for dataset in dataset_list:
                    with open(os.path.join(output_path, f'{dataset}_auc.json'), 'w') as f:
                        json.dump(auc_dict[dataset], f)
        
        if is_main_process():
            with open(output_file, 'a') as writer:
                writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
            writer.close()
        
        