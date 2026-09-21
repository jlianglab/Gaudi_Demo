import os
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import yaml
from scipy import interpolate
from PIL import Image
import matplotlib.pyplot as plt
import torch.distributed as dist

def plot(auc_dict, cycles, dataset_list, output_path):
    
    for order, dataset in enumerate(dataset_list):
        plt.figure(figsize=(10, 6))
        
        plt.plot(cycles, auc_dict[dataset]['student'][len(dataset_list)-1::len(dataset_list)], label='Student', marker='o', markersize=3)
        plt.plot(cycles, auc_dict[dataset]['teacher'][len(dataset_list)-1::len(dataset_list)], label='Teacher', marker='s', markersize=3)
        
        plt.xlabel('Cycle')
        plt.ylabel('AUC')
        plt.title('Ark %s Test AUC' % (dataset))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, f'{dataset}_student_teacher.png'), dpi=150)
        
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        n = len(dataset_list)
        
        cycles_focused = np.arange(1*(order+1)/n, len(cycles)+1/n, 1)
        focused_y = np.array(auc_dict[dataset]['student'][order::n])
        
        cycles_unfocused = np.arange(1/n, len(cycles)+1/n, 1/n)
        unfocused_y = np.array(auc_dict[dataset]['student'])
        
        color = 'tab:blue'
        
        plt.plot(cycles_focused, focused_y, label="Focused Training", linestyle='-', color=color)
        plt.plot(cycles_unfocused, unfocused_y, label="Unfocused Training", linestyle='-', color=color, alpha=0.5)
        
        focused_mask = (np.arange(len(cycles_unfocused)) % n) == order
        
        plt.scatter(cycles_unfocused[focused_mask], unfocused_y[focused_mask],
                    s=20, facecolors=color, edgecolors=color, zorder=3, alpha=1.0)
        
        plt.scatter(cycles_unfocused[~focused_mask], unfocused_y[~focused_mask],
                    s=20, facecolors='none', edgecolors=color, zorder=3, alpha=0.5)
        
        plt.xlabel('Cycle')
        plt.ylabel('AUC')
        plt.title('Ark %s Test AUC' % (dataset))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, f'{dataset}_student_focus.png'), dpi=150)
        
        plt.close()
    
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def cleanup():
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if args.device == 'cuda':
        # launched with torch.distributed.launch
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        # launched with submitit on a slurm cluster
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        # launched naively with `python main_dino.py`
        # we manually add MASTER_ADDR and MASTER_PORT to env variables
        elif torch.cuda.is_available():
            print('Will run the code on one GPU.')
            args.rank, args.gpu, args.world_size = 0, 0, 1
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
        else:
            print('Does not support training without GPU.')
            sys.exit(1)
        
        if args.world_size == 1:
            args.distributed = False
            return
        
        args.distributed = True
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

        torch.cuda.set_device(args.gpu)

    elif args.device == 'hpu':    
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
        args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()

        if args.world_size == 1:
            args.distributed = False
            return
        
        args.distributed = True
        
        dist.init_process_group('hccl', rank=args.rank, world_size=args.world_size)
    
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self, device):    
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def metric_AUROC(target, output, nb_classes=14):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        if np.any(target[:, i]):
            outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC


def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)

def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]


def save_image(input,idx):

    def disparity_normalization(disp):  # disp is an array in uint8 data type
        _min = np.amin(disp)
        _max = np.amax(disp)
        disp_norm = (disp - _min) * 255.0 / (_max - _min)
        return np.uint8(disp_norm)

    im = disparity_normalization(input)
    im = Image.fromarray(im)
    im.save("{}.jpeg".format(idx))

def save_snapshot(samples, masks, outputs, save_snapshot_path):
    bsz = samples.shape[0]
    
    snap_shot = torch.cat((samples[:,0,:,:],masks[:,0,:,:],outputs[:,0,:,:],masks[:,1,:,:],outputs[:,1,:,:],masks[:,2,:,:],outputs[:,2,:,:]), dim=2)
    l = []
    for b in range(bsz):
        l.append(snap_shot[b])
    snap_shot = torch.cat(l, dim=0)
    print("Saving snapshot..... Size:", snap_shot.size())
    save_image(snap_shot.cpu().numpy(), save_snapshot_path)

# --------------------------------------------------------
# Copy from DINO https://github.com/facebookresearch/dino
# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

# --------------------------------------------------------
# Copy from SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------
def load_pretrained_simmim(pretrained_weights, model):
    print(">>>>>>>>>> Fine-tuned from {pretrained_weights} ..........")
    checkpoint = torch.load(pretrained_weights, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        print('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        print('Detect non-pre-trained model, pass without doing anything.')
    print(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
    checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)
    # if args.model_name.startswith('swin'):
    #     print(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
    #     checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)
    # elif args.model_name.startswith('vit'):
    #     print(">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
    #     checkpoint = remap_pretrained_keys_vit(model, checkpoint_model)
    # else:
    #     raise NotImplementedError

    # msg = model.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    
    # del checkpoint
    # torch.cuda.empty_cache()
    # print(">>>>>>>>>> loaded successfully '{}'".format(pretrained_weights))
    


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print("Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print("{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model