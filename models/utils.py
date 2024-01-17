import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from compressai_dep import conv, deconv, GDN, ResidualBlock, ResidualBlockUpsample, ResidualBlockWithStride, MaskedConv2d

C1   = float(1.0)
C255 = float(255.0)

conv_k3s1 = lambda c_i, c_o: conv(c_i, c_o, stride=1, kernel_size=3)
lrelu = lambda *args, **kwargs: nn.LeakyReLU(inplace=False)
iGDN = lambda N: GDN(N, inverse=True)

cat = lambda *x_list: torch.cat(x_list, dim=1)

def ccat(x, y):
  if x is not None:
    return cat(x, y) if y is not None else x
  else:
    return y