from utils import AverageMeter, ProgressLogger, save_image, save_snapshot, get_world_size
import time
import torch
import torch.distributed as dist
from tqdm import tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    pass

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it, lazy_mode):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_cls = AverageMeter('Loss_'+dataset+' cls', ':.4e')
    losses_mse = AverageMeter('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    #print(momentum_schedule[it],it, coff)
    end = time.time()
    for i, (samples1, samples2, targets, _) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)
        
        feat_t, pred_t = teacher(samples2, use_head_n)
        feat_s, pred_s = model(samples1, use_head_n)
        loss_cls = criterion(pred_s, targets)
        loss_const = MSE(feat_s, feat_t)

        # outputs_t = teacher(samples2)
        # outputs_s = model(samples1)
        # loss_cls = criterion(outputs_s[use_head_n], targets)
        # loss_const = 0
        # for i in range(len(outputs_t)):
        #     loss_const += MSE(outputs_t[i], outputs_s[i])
        
        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()

        if lazy_mode:
            htcore.mark_step()

        optimizer.step()

        if lazy_mode:
            htcore.mark_step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it, lazy_mode)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it, lazy_mode)
        it += 1

def ema_update_teacher(model, teacher, momentum_schedule, it, lazy_mode):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    
    if lazy_mode:
        htcore.mark_step()


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset, lazy_mode, distributed):
    model.eval()

    with torch.no_grad():
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets, _) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            if lazy_mode:
                htcore.mark_step()

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    if distributed:
        losses.all_reduce(device)
    
    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, lazy_mode, multiclass = False): 
       
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)
    idx_test = torch.LongTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets, indices) in enumerate(tqdm(data_loader_test)):
            targets = targets.to(device)
            indices = indices.to(device)
            y_test = torch.cat((y_test, targets), 0)
            idx_test = torch.cat((idx_test, indices), 0)
            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

            _, out = model(varInput, use_head_n)
            
            if lazy_mode:
                htcore.mark_step()
            
            if multiclass:
                out = torch.softmax(out,dim = 1)
            else:
                out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test, idx_test

def test_gather(y_test, p_test, idx_test, dataset_len, device):
    world_size = get_world_size()
    y_list = [torch.zeros_like(y_test) for _ in range(world_size)]
    p_list = [torch.zeros_like(p_test) for _ in range(world_size)]
    idx_list = [torch.zeros_like(idx_test) for _ in range(world_size)]

    dist.all_gather(y_list, y_test)
    dist.all_gather(p_list, p_test)
    dist.all_gather(idx_list, idx_test)
    
    y_all = torch.cat(y_list).to(device)
    p_all = torch.cat(p_list).to(device)
    idx_all = torch.cat(idx_list).to(device)
    
    y_final = torch.empty(dataset_len, *y_all.shape[1:], dtype=y_all.dtype).to(device)
    p_final = torch.empty(dataset_len, *p_all.shape[1:], dtype=p_all.dtype).to(device)
    y_final[idx_all] = y_all
    p_final[idx_all] = p_all

    return y_final, p_final
    