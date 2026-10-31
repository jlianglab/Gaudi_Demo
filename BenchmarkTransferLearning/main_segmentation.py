import os
import sys
import shutil
import time
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int, get_config
from dataloader import  *
import argparse

import torch
from engine import segmentation_engine
from utils import torch_dice_coef_loss

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Command line arguments for segmentation target tasks.')
    parser.add_argument('--config', help='config file',
                        default=None)
    parser.add_argument('--data_set', help='target dataset',
                        default=None)
    parser.add_argument('--train_batch_size', help='train batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', help='test batch size', default=32, type=int)
    parser.add_argument('--epochs', help='number of epochs', default=500, type=int)
    parser.add_argument('--train_num_workers', help='train num of parallel workers for data loader', default=2,
                        type=int)
    parser.add_argument('--test_num_workers', help='test num of parallel workers for data loader', default=2, type=int)
    parser.add_argument('--distributed', help='whether to use distributed or not', dest='distributed',
                        action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--learning_rate', help='learning rate', default=0.0002, type=float)
    parser.add_argument('--mode', help='train|test', default='train')
    parser.add_argument('--backbone', help='encoder backbone', default='convnext_base')
    parser.add_argument('--arch', help='segmentation network architecture', default='convnext_upernet')
    parser.add_argument('--proxy_dir', help='path to pre-trained model', default=None)
    parser.add_argument('--device', help='cuda|cpu', default="cuda")
    parser.add_argument('--run', type=int, default='1', help='trial number')
    parser.add_argument('--init', help='None (random) |ImageNet |or other pre-trained methods', default=None)
    parser.add_argument('--normalization', help='imagenet|None', default="imagenet")
    parser.add_argument('--activate', help='activation', default="sigmoid")
    parser.add_argument('--patience', type=int, default=50, help='num of patient epochesr')
    parser.add_argument('--img_size', type=int, default=224, help='input image resolution')
    parser.add_argument('--num_classes', dest='num_classes', default=1, type=int)
    parser.add_argument('--optimizer', dest='optimizer', help="adam | sgd | adamw", default="adamw")
    parser.add_argument("--save_dir", dest="save_dir", help="checkpoints and outputs save directory", default="./benchmark_transfer")


    args = parser.parse_args()
    return args

def main(args):
    print(args)
    assert args.config is not None
    assert args.data_set is not None

    if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
        assert args.proxy_dir is not None

    if args.init is not None:
        model_path = os.path.join(args.save_dir,"Models/Segmentation", args.data_set, args.arch, args.backbone, args.init,str(args.run))
    else:
        model_path = os.path.join(args.save_dir,"Models/Segmentation", args.data_set, args.arch, args.backbone, "random",str(args.run))

    args.lazy_mode = os.getenv('PT_HPU_LAZY_MODE', '0') == '1'

    dataset_config = get_config(args.config)
    args.train_data_dir = dataset_config[args.data_set]['train_data_dir']
    args.valid_data_dir = dataset_config[args.data_set]['valid_data_dir']
    args.test_data_dir = dataset_config[args.data_set]['test_data_dir']
    args.train_mask_dir = dataset_config[args.data_set]['train_mask_dir']
    args.valid_mask_dir = dataset_config[args.data_set]['valid_mask_dir']
    args.test_mask_dir = dataset_config[args.data_set]['test_mask_dir']

    if args.data_set == "SCR-Heart" or args.data_set == "SCR-Clavicle":
        args.train_file_path = dataset_config[args.data_set]['train_file_path']
        args.valid_file_path = dataset_config[args.data_set]['valid_file_path']
        args.test_file_path = dataset_config[args.data_set]['test_file_path']

    if args.data_set == "Montgomery":
        dataset_train = MontgomeryDataset(args.train_data_dir,args.train_mask_dir,transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = MontgomeryDataset(args.valid_data_dir,args.valid_mask_dir,transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = MontgomeryDataset(args.test_data_dir,args.test_mask_dir,transforms=None, normalization=args.normalization)
        criterion = torch_dice_coef_loss
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    elif args.data_set == "DRIVE":
        dataset_train = DriveDataset(args.train_data_dir,args.train_mask_dir)
        dataset_val = DriveDataset(args.valid_data_dir,args.valid_mask_dir)
        dataset_test = DriveDataset(args.test_data_dir,args.test_mask_dir)
        criterion = torch.nn.BCELoss()
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    elif args.data_set == "SIIM_PNE": #Pneumothorax segmentation
        dataset_train = PNEDataset(args.train_data_dir, args.train_mask_dir,
                                          transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = PNEDataset(args.valid_data_dir, args.valid_mask_dir,
                                        transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = PNEDataset(args.test_data_dir, args.test_mask_dir, transforms=None,
                                         normalization=args.normalization)
        criterion = torch_dice_coef_loss
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    elif args.data_set == "SCR-Heart" or args.data_set == "SCR-Clavicle": #Heart & Clavicle segmentation
        dataset_train = SCRDataset(args.train_data_dir, args.train_mask_dir,args.train_file_path,
                                              transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = SCRDataset(args.valid_data_dir, args.valid_mask_dir,args.valid_file_path,
                                            transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = SCRDataset(args.test_data_dir, args.test_mask_dir,args.test_file_path, transforms=None,
                                             normalization=args.normalization)
        criterion = torch_dice_coef_loss
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    elif args.data_set == "USBreast": #Breast cancer segmentation
        dataset_train = SCRDataset(args.train_data_dir, args.train_mask_dir,args.train_file_path,
                                              transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = SCRDataset(args.valid_data_dir, args.valid_mask_dir,args.valid_file_path,
                                            transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = SCRDataset(args.test_data_dir, args.test_mask_dir,args.test_file_path, transforms=None,
                                             normalization=args.normalization)
        criterion = torch_dice_coef_loss
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)

