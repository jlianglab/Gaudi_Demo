from utils import AverageMeter, ProgressLogger
from models import ClassificationNet, build_classification_model
import time
import torch
from tqdm import tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    pass

def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch, args):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in enumerate(data_loader_train):
    samples, targets = samples.float().to(device, non_blocking=True), targets.float().to(device, non_blocking=True)
    
    outputs = model(samples)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    
    if args.lazy_mode:
      htcore.mark_step()
    
    optimizer.step()
    
    if args.lazy_mode:
      htcore.mark_step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)


def evaluate(data_loader_val, device, model, criterion, args):
  model.eval()

  with torch.no_grad():
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      samples, targets = samples.float().to(device, non_blocking=True), targets.float().to(device, non_blocking=True)

      outputs = model(samples)
      loss = criterion(outputs, targets)
      
      if args.lazy_mode:
        htcore.mark_step()
      
      losses.update(loss.item(), samples.size(0))
      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

  losses.all_reduce(device)

  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  model = build_classification_model(args)
  print(model)
    
  model.to(device)

  modelCheckpoint = torch.load(checkpoint, weights_only=False)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
      state_dict[k[len("_orig_mod."):]] = state_dict[k]
      del state_dict[k]
     
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))

  if args.torch_compile:
    torch.compile(model, backend="hpu_backend")

  model.eval()

  y_test = torch.FloatTensor().to(device)
  p_test = torch.FloatTensor().to(device)

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.to(device)
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

      out = model(varInput)

      if args.lazy_mode:
        htcore.mark_step()
      
      if args.data_set == "RSNAPneumonia":
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test


