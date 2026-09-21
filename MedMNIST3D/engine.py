import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from medmnist.info import INFO
from medmnist.evaluator import Evaluator
from models import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from tqdm import trange
from utils import Transform3D, model_to_syncbn
import torch.distributed as dist
from utils import utils
import yaml

try:
    import habana_frameworks.torch.core as htcore
except:
    pass

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def main(args, data_flag, output_root, data_root, num_epochs, gpu_ids, device, batch_size, workers, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run, compile, lazy_mode):
    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    device = torch.device(args.device) 
    
    mode = args.device
    
    if mode == 'hpu':
        if lazy_mode:
            mode += '_lazy'
        else:
            mode += '_eager'
    
    output_root = os.path.join(output_root, data_flag, f'{mode}_compile_{compile}', time.strftime("%y%m%d_%H%M%S"))
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()

    train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb, size=size, root=data_root)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb, size=size, root=data_root)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb, size=size, root=data_root)
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=workers)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers)

    print('==> Building and training model...')

    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    if model_flag in ['resnet18', 'resnet50']:
        if conv=='ACSConv':
            model = model_to_syncbn(ACSConverter(model))
        if conv=='Conv2_5d':
            model = model_to_syncbn(Conv2_5dConverter(model))
        if conv=='Conv3d':
            if pretrained_3d == 'i3d':
                model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
            else:
                model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))

    model = model.to(device)

    if compile:
        model = torch.compile(model,backend="hpu_backend")
    
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size, root=data_root)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size, root=data_root)

    criterion = nn.CrossEntropyLoss()

    if num_epochs == 0:
        return
    
    print(f'Params: {len(list(model.parameters()))}')
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

    fp16_scaler = None
    if args.device == 'cuda':
        fp16_scaler = torch.amp.GradScaler('cuda')
    
    for epoch in trange(num_epochs):
        
        train_loss = train(model, train_loader, criterion, optimizer, device, writer, lazy_mode, fp16_scaler)
        
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run, lazy_mode, fp16_scaler)
        
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
    test_metrics = test(best_model, test_evaluator, test_loader, criterion, device, run, lazy_mode, output_root)

    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + test_log + '\n'
    print(log)
    
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)        
            
    writer.close()


def train(model, train_loader, criterion, optimizer, device, writer, lazy_mode, fp16_scaler):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.float().to(device)
        targets = torch.squeeze(targets, 1).long().to(device)
        
        mp_type = torch.float16 if args.device == 'cuda' else torch.bfloat16
        with torch.autocast(device_type=args.device, dtype=mp_type):
            outputs = model(inputs)
            loss = criterion(outputs, targets) #still runs in fp32
        
        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1
        optimizer.zero_grad()

        if fp16_scaler is None:
            loss.backward()

            if lazy_mode:
                htcore.mark_step()
            
            optimizer.step()

            if lazy_mode:
                htcore.mark_step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        if batch_idx % 50 == 0:
            print(f'done with iter {batch_idx}')
        
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, criterion, device, run, lazy_mode, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
        
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

            if lazy_mode:
                htcore.mark_step()
            
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)\
            
            if batch_idx % 50 == 0:
                print(f'done with iter {batch_idx}')

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='organmnist3d',
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
                        default=28,
                        help='the image size of the dataset, 28 or 64, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--device',
                        default='hpu',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--conv',
                        default='ACSConv',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone, resnet18/resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
                        distributed training""")
    parser.add_argument('--torch_compile',
                        help='Whether to use torch compile',
                        action="store_true")
    parser.add_argument('--workers',
                        default=0,
                        type=int)


    args = parser.parse_args()
    data_flag = args.data_flag
    
    dataset_config = get_config(args.config)
    output_root = dataset_config['medmnist']['output_root']
    data_root = dataset_config['medmnist']['data_root']

    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    device = args.device
    workers = args.workers
    conv = args.conv
    pretrained_3d = args.pretrained_3d
    download = args.download
    model_flag = args.model_flag
    as_rgb = args.as_rgb
    model_path = args.model_path
    shape_transform = args.shape_transform
    run = args.run
    compile = args.torch_compile
    lazy_mode = os.getenv('PT_HPU_LAZY_MODE', '0') == '1'
    print(f'compile: {compile}')
    print(f'lazy mode: {lazy_mode}')
    main(args, data_flag, output_root, data_root, num_epochs, gpu_ids, device, batch_size, workers, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run, compile, lazy_mode)
