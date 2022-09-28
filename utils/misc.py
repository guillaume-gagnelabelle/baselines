import logging

import torch
from torch import nn
import random
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime


logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def create_loss_fn(args):
    return nn.CrossEntropyLoss().to(args.device)


def xent(distr, labels, reduction='mean'):
  logs = - torch.sum(torch.log(torch.gather(distr, 1, labels.unsqueeze(1))))
  if reduction == 'mean':
    return logs / labels.size(0)
  if reduction == 'sum':
    return logs

def accuracy_fixmatch(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_pham(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def console_log(args, main_train=True):
  print("-------------- seed: %d --- t: %d --------------" % (args.seed, args.t))
  print(datetime.now())
  print("nb_lbl: %s" % (args.num_labeled))
  if main_train: print("Train     Loss : %.4f" % args.logs['train_loss'][args.t - 1])
  print("Test      Loss : %.9f" % args.logs['test_loss'][args.t])
  print("Val. Test Loss : %.9f" % args.logs['val_test_loss'][args.t])
  print("Val. Train Loss: %.9f" % args.logs['val_train_loss'][args.t])
  print("Test       Acc : %.2f" % args.logs['test_acc'][args.t])
  print("Val. Test  Acc : %.2f" % args.logs['val_test_acc'][args.t])
  print("Val. Train Acc : %.2f \n" % args.logs['val_train_acc'][args.t])


def trim(probs):
  new_probs = probs.clone()
  new_probs[probs < 1e-9] = 1e-9
  new_probs[probs > 1e9] = 1e9
  return new_probs


class ClassifierModel(nn.Module):
  def __init__(self, L, C):
    super().__init__()
    self.pred = nn.Sequential(nn.Linear(L, C))

  def forward(self, d_x):
    return self.pred(d_x)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
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
