import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def get_enc_sz(in_dim):
  if isinstance(in_dim, int):
    enc_sz = 32
  else:
    enc_sz = 256 # 512 #

  return enc_sz


def get_enc(in_dim, L):
  enc_sz = get_enc_sz(in_dim)
  if isinstance(in_dim, int):
    return nn.Sequential(
        nn.Linear(in_dim, enc_sz),
        nn.ReLU(),
        nn.Linear(enc_sz, L))

  else:
    return nn.Sequential(
        nn.Conv2d(in_dim[0], 32, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.MaxPool2d(2, 2), # 14, 14
        nn.Flatten(start_dim=1),
        nn.Linear(64 * 14 * 14, enc_sz),
        nn.BatchNorm1d(enc_sz),

        nn.ReLU(),
        nn.Linear(enc_sz, L),
        nn.BatchNorm1d(L),
    )


class InferenceModel(nn.Module):
  def __init__(self, in_dim, L, C):
    super().__init__()

    self.L = L
    self.pred = get_enc(in_dim, L)
    self.cls = ClassifierModel(L, C)


  def forward(self, d_x):
    pred = self.pred(d_x)
    pred = self.cls(pred)
    return pred

class TargetModel(nn.Module):
  def __init__(self, in_dim, L):
    super().__init__()

    self.L = L
    self.feat_space = get_enc(in_dim, L)


  def forward(self, d_x):
    encoding = self.feat_space(d_x)
    encoding = F.normalize(encoding)

    return encoding


class ClassifierModel(nn.Module):
  def __init__(self, L, C):
    super().__init__()

    self.pred = nn.Sequential(
      nn.Linear(L, C), # not softmaxed
    )

  def forward(self, d_x):
    return self.pred(d_x)


def build_InferenceModel(args):
    return InferenceModel(args.in_dim, args.L, args.num_classes)
