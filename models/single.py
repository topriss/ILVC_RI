from collections import defaultdict

from .quant import ste, aun
from .entropy_models import FullyFactorized
from .msgau_cdf import MSGau_CDF
from .compression_models import CompressionBasic
from .utils import *

class GO(object):
  def __init__(self, name=''):
    self.name = name
  def __enter__(self):
    pass
  def __exit__(self, type, value, traceback):
    pass

class single(CompressionBasic):

  def __init__(self, N, M, **kwargs):
    super().__init__()
    C = [N, N, N]
    self.g_a = nn.Sequential(
                 conv(3,    C[0]),
      GDN(C[0]), conv(C[0], C[1]),
      GDN(C[1]), conv(C[1], C[2]),
      GDN(C[2]), conv(C[2], M),
    )
    self.g_s = nn.Sequential(
                  deconv(M,    C[2]),
      iGDN(C[2]), deconv(C[2], C[1]),
      iGDN(C[1]), deconv(C[1], C[0]),
      iGDN(C[0]), deconv(C[0], 3),
    )
    self.h_a = nn.Sequential(
               conv_k3s1(M, M),
      lrelu(), conv     (M, M),
      lrelu(), conv     (M, N),
    )
    self.h_s_mu = nn.Sequential(
               deconv   (N, M),
      lrelu(), deconv   (M, M),
      lrelu(), conv_k3s1(M, M),
    )
    self.h_s_sigma = nn.Sequential(
               deconv   (N, M),
      lrelu(), deconv   (M, M),
      lrelu(), conv_k3s1(M, M),
    )
    self.z_EM = FullyFactorized(N)
    self.y_EM = MSGau_CDF      ()
  
  def forward(self, x):
    y = self.g_a(x)
    z = self.h_a(y)

    with GO():
      z_hat = ste(z)
      z_ll  = self.z_EM(aun(z))
    mu    = self.h_s_mu   (z_hat)
    sigma = self.h_s_sigma(z_hat)
    with GO():
      y_hat = self.y_EM.sym_bound(ste(y))
      y_ll  = self.y_EM(self.y_EM.sym_bound(aun(y)), sigma, mu)
    
    x_hat = self.g_s(y_hat)

    ret = { 
      'likelihoods': {'z': z_ll, 'y': y_ll},
      'x_hat' : x_hat,
    } 
    return ret
  
  def compress(self, x):
    y = self.g_a(x)
    z = self.h_a(y)

    with GO():
      z_hat = ste(z)
      z_str = self.z_EM.compress(z_hat, self.z_EM.build_indexes(z.size()))
    mu    = self.h_s_mu   (z_hat)
    sigma = self.h_s_sigma(z_hat)
    with GO():
      y_hat = self.y_EM.sym_bound(ste(y))
      y_str = self.y_EM.compress(y_hat, sigma, mu)

    ret = {
      'strings': {'z': z_str, 'y': y_str}, 
      'shape': z.size() 
    }
    return ret

  def decompress(self, strings, size):
    for p in self.parameters():
      dtype = p.dtype
      break
    with GO():
      z_hat = self.z_EM.decompress(strings['z'], self.z_EM.build_indexes(size), dtype=dtype)
    mu    = self.h_s_mu   (z_hat)
    sigma = self.h_s_sigma(z_hat)
    with GO():
      y_hat = self.y_EM.decompress(strings['y'], sigma, mu)

    x_hat = self.g_s(y_hat)

    ret = { 'x_hat': x_hat }
    return ret