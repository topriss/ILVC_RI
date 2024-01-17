from functools import partial

from torch.autograd.profiler import record_function

import geotorch

from .utils import *
from .single import single

def unfold(x, K):
  N, C, H, W = x.size()
  return F.unfold(x, [K, K], stride=[K, K]).reshape(N, C * K**2, H//K, W//K).permute(0, 2, 3, 1)[..., None]

def fold(x, K):
  N, nh, nw, ckk = x.squeeze(-1).size()
  C = ckk // K**2
  return x.reshape(N, nh, nw, C, K, K).permute(0, 3, 1, 4, 2, 5).reshape(N, C, nh * K, nw * K)

class psingle(single):

  def __init__(self, N, M, **kwargs):
    super().__init__(N, M, **kwargs)
    self.g_a = None
    self.g_s = None
    C = [int(_c) for _c in kwargs.get('C', [10, 38, 150])]
    P = kwargs.get('pk', 1)
    cnnk = kwargs.get('cnnk', 5)
    c_nn = lambda c: nn.Conv2d(c, c, cnnk, padding=cnnk//2, stride=1)
    g = [
      [ offset() ],

      [ pconv2d(3,    2*P, C[0], 1*P, **kwargs) ],
      [ channel_coup(C[0], c_nn), channel_coup(C[0], c_nn, shuffle=True) ],
      [ channel_coup(C[0], GDN),  channel_coup(C[0], GDN,  shuffle=True) ],

      [ pconv2d(C[0], 2*P, C[1], 1*P, **kwargs) ],
      [ channel_coup(C[1], c_nn), channel_coup(C[1], c_nn, shuffle=True) ],
      [ channel_coup(C[1], GDN),  channel_coup(C[1], GDN,  shuffle=True) ],

      [ pconv2d(C[1], 2*P, C[2], 1*P, **kwargs) ],
      [ channel_coup(C[2], c_nn), channel_coup(C[2], c_nn, shuffle=True) ],
      [ channel_coup(C[2], GDN),  channel_coup(C[2], GDN,  shuffle=True) ],

      [ pconv2d(C[2], 2*P, M,    1*P, **kwargs) ],
      [ channel_coup(M,    c_nn), channel_coup(M,    c_nn, shuffle=True) ],
    ]
    self.g = nn.ModuleList([__g for _g in g if _g for __g in _g])
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data = m.weight.data * 0.1
        m.bias  .data = m.bias  .data * 0
  
  def g_a(self, x):
    for m in self.g:
      x = m(x)
    return x
  
  def g_s(self, x):
    for m in self.g[::-1]:
      x = m.forward_p(x)
    return x

class pconv2d(nn.Module):
  # (ci, ki, ki) <-> (co, ko, ko)
  # stride === kernel size
  def __init__(self, ci, ki, co, ko, **kwargs):
    super().__init__()
    self.ci = ci
    self.ki = ki
    self.co = co
    self.ko = ko
    self.A = A_lazy(co * ko**2, ci * ki**2)
    self.zero_path = kwargs.get('zero_path', True)
    if self.zero_path:
      zn = kwargs.get('ZN', 0)
      if co * ko**2 < ci * ki**2: # surjective A
        self.zp_func = ResidualBlockUpsample(co, ci) if zn == 0 else nn.Sequential(ResidualBlockUpsample(co, zn), ResidualBlock(zn, ci))
      elif co * ko**2 > ci * ki**2: # surjective A+
        self.zp_func = ResidualBlockWithStride(ci, co) if zn == 0 else nn.Sequential(ResidualBlockWithStride(ci, zn), ResidualBlock(zn, co))
    self.bias = Parameter(torch.randn(co)) if kwargs.get('bias', True) else None
  
  def cal_y(self, x):
    with record_function('### range_path'):
      y = fold(self.A.w @ unfold(x, self.ki), self.ko)
      if self.bias is not None:
        y = y + self.bias.view(1, -1, *[1 for _ in range(y.ndim - 2)])
    return y

  def cal_x(self, y):
    with record_function('### range_path'):
      if self.bias is not None:
        y = y - self.bias.view(1, -1, *[1 for _ in range(y.ndim - 2)])
      x = fold(self.A.w_pinv @ unfold(y, self.ko), self.ki)
    return x

  def cal_y_zp(self, x):
    with record_function('### zero_path'):
      if (self.zero_path) and (self.co * self.ko**2 > self.ci * self.ki**2):
        y_zp = self.zp_func(x)
        y_zp = y_zp - fold(self.A.w_wp @ unfold(y_zp, self.ko), self.ko)
      else:
        y_zp = 0
    return y_zp
  
  def cal_x_zp(self, y):
    with record_function('### zero_path'):
      if (self.zero_path) and (self.co * self.ko**2 < self.ci * self.ki**2):
        x_zp = self.zp_func(y)
        x_zp = x_zp - fold(self.A.wp_w @ unfold(x_zp, self.ki), self.ki)
      else:
        x_zp = 0
    return x_zp

  def forward(self, x):
    return self.cal_y(x) + self.cal_y_zp(x)
  
  def forward_p(self, y):
    return self.cal_x(y) + self.cal_x_zp(y)

class A_lazy(nn.Module):
  def __init__(self, O, I, **kwargs):
    super().__init__()
    self.rank = min(O, I)
    self.U = Q_lazy(O, **kwargs)
    self.V = Q_lazy(I, **kwargs)
    geotorch.orthogonal(self.U, '_Q')
    geotorch.orthogonal(self.V, '_Q')
    self._S = Parameter(torch.zeros(self.rank))
  
  @property
  def S(self):
    return (self._S.sigmoid() * 2 - 1).exp()
  
  @property
  def w(self):
    return self.U.Q[:, :self.rank] @ torch.diag(self.S)     @ self.V.Q[:, :self.rank].T
  
  @property
  def w_pinv(self):
    return self.V.Q[:, :self.rank] @ torch.diag(1.0/self.S) @ self.U.Q[:, :self.rank].T

  @property
  def w_wp(self):
    return self.U.Q[:, :self.rank] @ self.U.Q[:, :self.rank].T

  @property
  def wp_w(self):
    return self.V.Q[:, :self.rank] @ self.V.Q[:, :self.rank].T

class Q_lazy(nn.Module):
  def __init__(self, n, **kwargs):
    super().__init__()
    self._Q = Parameter(torch.randn(n, n))
  
  @property
  def Q(self):
    return self._Q

class channel_coup(nn.Module):
  def __init__(self, C, fclass, shuffle=False):
    super().__init__()
    assert C % 2 == 0
    self._fs = fclass(C // 2)
    self.ft = fclass(C // 2)
    self.fs = lambda x: (self._fs(x).sigmoid() * 2 - 1).exp() # [1/e^2, e^2]
    self.shuffle = channel_shuffle if shuffle else lambda x: x
  
  def forward(self, x):
    x1, x2 = self.shuffle(x).chunk(2, 1)
    y2 = x2 * self.fs(x1) + self.ft(x1)
    return cat(x1, y2)
  
  def forward_p(self, y):
    y1, y2 = y.chunk(2, 1)
    x2 = (y2 - self.ft(y1)) / self.fs(y1)
    return self.shuffle(cat(y1, x2))

def channel_shuffle(x):
  x1, x2 = x.chunk(2, 1)
  return cat(x2, x1)

class offset(nn.Module):
  def __init__(self, o=0.5):
    super().__init__()
    self.o = o
  
  def forward(self, x):
    return x - self.o
  
  def forward_p(self, y):
    return y + self.o