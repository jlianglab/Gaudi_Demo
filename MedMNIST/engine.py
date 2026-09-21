import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import random
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist.info import INFO
from medmnist.evaluator import Evaluator
from medmnist.dataset import (PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST,
                                  BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
                                  OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D)
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from torchvision.models import resnet18, resnet50
from tqdm import trange
import yaml

try:
    import habana_frameworks.torch.core as htcore
except:
    pass

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(data_flag, output_root, data_root, num_epochs, gpu_ids, device, batch_size, workers, size, download, model_flag, resize, as_rgb, model_path, trials, runs, fix_seeds, determinism, seed, multiple_seeds, fix_at_beginning, lazy_mode, compile):
    start_time = time.time()
    print(f'fix_seeds: {fix_seeds}')
    print(f'determinism: {determinism}')
    
    g = torch.Generator()

    if fix_seeds and fix_at_beginning:
        fix_random_seeds(seed)
        g.manual_seed(seed)
        
    if determinism:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = False
    
    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = globals()[info['python_class']]

    device = torch.device(device) 
    
    mode = args.device
    
    if mode == 'hpu':
        if lazy_mode:
            mode += '_lazy'
        else:
            mode += '_eager'

    if fix_seeds and multiple_seeds:
        results_root = os.path.join(output_root, f'fixed_seed_{fix_seeds}_beginning_{fix_at_beginning}_determinism_{determinism}', f'{mode}_compile_{compile}', 'multiple_seeds', data_flag)
        output_root = os.path.join(results_root, time.strftime("%y%m%d_%H%M%S"))
    elif fix_seeds and runs > 1:
        results_root = os.path.join(output_root, f'fixed_seed_{fix_seeds}_beginning_{fix_at_beginning}_determinism_{determinism}', f'{mode}_compile_{compile}', 'single_runs', data_flag)
        output_root = os.path.join(results_root, time.strftime("%y%m%d_%H%M%S"))
    else:
        results_root = os.path.join(output_root, f'fixed_seed_{fix_seeds}_beginning_{fix_at_beginning}_determinism_{determinism}', f'{mode}_compile_{compile}', data_flag)
        output_root = os.path.join(results_root, time.strftime("%y%m%d_%H%M%S"))
    
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    print(output_root)
    
    print('==> Preparing data...')

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    
    for trial in range(1, trials+1):
        
        if fix_seeds and not fix_at_beginning:
            fix_random_seeds(seed)
            g.manual_seed(seed)
        
        run = "run_"+str(trial)
        
        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size, root=data_root)
        val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size, root=data_root)
        test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size, root=data_root)
    
        
        train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    num_workers=workers)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=workers)
        val_loader = data.DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=workers)
        test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=workers)
        
        if fix_seeds:
            train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    num_workers=workers,
                                    drop_last=True)
            train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        num_workers=workers)
            val_loader = data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        num_workers=workers)
            test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        num_workers=workers)
        
        print('==> Building and training model...')
        
        
        if model_flag == 'resnet18':
            model = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
        elif model_flag == 'resnet50':
            model = resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
        else:
            raise NotImplementedError
    
        model = model.to(device)

        if compile:
            model = torch.compile(model,backend="hpu_backend")
        
        train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size, root=data_root)
        val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size, root=data_root)
        test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size, root=data_root)
    
        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
            train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, output_root)
            val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, output_root)
            test_metrics = test(model, test_evaluator, test_loader, task, criterion, device, run, output_root)
    
            print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1]*100, train_metrics[2]*100) + \
                  'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1]*100, val_metrics[2]*100) + \
                  'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1]*100, test_metrics[2])*100)
        
        if num_epochs == 0:
            return
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
        logs = ['loss', 'auc', 'acc']
        train_logs = ['train_'+log for log in logs]
        val_logs = ['val_'+log for log in logs]
        test_logs = ['test_'+log for log in logs]
        log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
        
        writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))
    
        best_auc = 0
        best_epoch = 0

        best_model = deepcopy(model)

        global iteration
        iteration = 0
        
        for epoch in trange(num_epochs):        
            train_loss = train(model, train_loader, task, criterion, optimizer, device, writer, lazy_mode)
            
            val_metrics = test(model, val_evaluator, val_loader, task, criterion, device, run, lazy_mode)
            
            scheduler.step()
            
            for i, key in enumerate(val_logs):
                log_dict[key] = val_metrics[i]
    
            for key, value in log_dict.items():
                writer.add_scalar(key, value, epoch)
                
            cur_auc = val_metrics[1]
            if cur_auc > best_auc:
                best_epoch = epoch
                best_auc = cur_auc

                best_model = deepcopy(model)

                print('cur_best_auc:', best_auc)
                print('cur_best_epoch', best_epoch)
        
        state = {
            'net': best_model.cpu().state_dict(),
        }
        
        path = os.path.join(output_root, 'best_model.pth')

        torch.save(state, path)
        
        best_model = best_model.to(device)
        
        test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device, run, lazy_mode, output_root)

        test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1]*100, test_metrics[2]*100)
        auc_log = '%.5f\n' % (test_metrics[1]*100)
        acc_log = '%.5f\n' % (test_metrics[2]*100)
        
        log = '%s\n' % (data_flag) + test_log
        
        print(log)
                
        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log) 
        
        if runs > 1:
            with open(os.path.join(output_root, '%s_auc.txt' % (data_flag)), 'a') as f:
                f.write(auc_log)
            with open(os.path.join(output_root, '%s_acc.txt' % (data_flag)), 'a') as f:
                f.write(acc_log)
            
        
        with open(os.path.join(results_root, '%s_auc.txt' % (data_flag)), 'a') as f:
            f.write(auc_log)
        
        with open(os.path.join(results_root, '%s_acc.txt' % (data_flag)), 'a') as f:
            f.write(acc_log) 
        
        writer.close()
    
    run_time = 'Run time: %.2f\n' % (time.time()-start_time)
    
    with open(os.path.join(results_root, '%s_runtime.txt' % (data_flag)), 'a') as f:
        f.write(run_time)


def train(model, train_loader, task, criterion, optimizer, device, writer, lazy_mode):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.float().to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1
        
        loss.backward()
        
        if lazy_mode:
            htcore.mark_step()
        
        optimizer.step()
        
        if lazy_mode:
            htcore.mark_step()
        
        if batch_idx % 100 == 0:
            print(f'done with iter {batch_idx}')
    
    epoch_loss = sum(total_loss)/len(total_loss)
    
    return epoch_loss
    
def test(model, evaluator, data_loader, task, criterion, device, run, lazy_mode, save_folder=None):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.float().to(device)
                loss = criterion(outputs, targets)
                
                if lazy_mode:
                    htcore.mark_step()
                
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                
                if lazy_mode:
                    htcore.mark_step()
                
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            
            if batch_idx % 100 == 0:
                print(f'done with iter {batch_idx}')

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--config',
                        default=None,
                        help='dataset config',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=224,
                        help='the image size of the dataset, 28 or 64 or 128 or 224, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--device',
                        default='hpu',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--workers',
                        default=8,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--trials',
                        default=1,
                        help='total trials to run',
                        type=int)
    parser.add_argument('--runs',
                        default=1,
                        help='independent runs',
                        type=int)                        
    parser.add_argument('--fix_seeds',
                        help='run with deterministic torch algorithms',
                        action="store_true")                    
    parser.add_argument('--determinism',
                        help='run with deterministic torch algorithms',
                        action="store_true")
    parser.add_argument('--seed',
                        default=0,
                        help='fixed random seed',
                        type=int)
    parser.add_argument('--multiple_seeds',
                        help='trials with different fixed seeds',
                        action="store_true")
    parser.add_argument('--fix_at_beginning',
                        help='only fix the seed at the beginning of the script',
                        action="store_true")
    parser.add_argument('--torch_compile',
                        help='Whether to use torch compile',
                        action="store_true")


    args = parser.parse_args()
    data_flag = args.data_flag

    dataset_config = get_config(args.config)
    output_root = dataset_config['medmnist']['output_root']
    data_root = dataset_config['medmnist']['data_root']

    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    device = args.device
    batch_size = args.batch_size
    workers = args.workers
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    model_path = args.model_path
    trials = args.trials
    runs = args.runs
    fix_seeds = args.fix_seeds
    determinism = args.determinism
    seed = args.seed
    multiple_seeds = args.multiple_seeds
    fix_at_beginning = args.fix_at_beginning
    compile = args.torch_compile
    lazy_mode = os.getenv('PT_HPU_LAZY_MODE', '0') == '1'
    print(lazy_mode)
    
    for run in range(runs):
        main(data_flag, output_root, data_root, num_epochs, gpu_ids, device, batch_size, workers, size, download, model_flag, resize, as_rgb, model_path, trials, runs, fix_seeds, determinism, seed, multiple_seeds, fix_at_beginning, lazy_mode, compile)