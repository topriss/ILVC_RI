# mean-scale Gaussian EM with element-wise CDF
import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaecl
from compressai_dep import LowerBound, UpperBound

HALF = float(0.5)
SCALE_UPPER_BOUND = False
PRINT_BITS = False

def pmf2cdf(pmf):
  assert (pmf >= 0).all()
  assert (pmf.max(dim=1)[0] > 0).all() # at least one pmf should > 1
  cdf = pmf.cumsum(dim=1)
  cdf = ( cdf / cdf[:, -1][:, None] * (2**16 - cdf.size(1)) ).round() # pmf to cdf avoid zero-probability
  cdf = cdf + torch.arange(cdf.size(1), device=cdf.device)[None, :] + 1
  cdf0 = torch.zeros(cdf.size(0), 1, dtype=cdf.dtype, device=cdf.device)
  cdf = torch.cat([ cdf0, cdf ], dim=1) # fixed 0 at beginning of cdf
  # assert (cdf.diff(dim=1) >= 1).all()
  # assert (cdf[:, 0]  == 0    ).all()
  # assert (cdf[:, -1] == 65536).all()
  return cdf

class MSGau_CDF(nn.Module):
  def __init__(self, 
    *args,
    sym_min=-128, sym_max=127, scale_bound=0.11, scale_upper_bound=256.0, ll_bound=0.5**16,
    **kwargs,
  ):
    super().__init__()
    assert sym_min + 1 < sym_max
    assert scale_bound > 0
    assert ll_bound > 0
    self.sym_min = int(sym_min)
    self.sym_max = int(sym_max)
    self.sym_lower_bound = LowerBound(self.sym_min)
    self.sym_upper_bound = UpperBound(self.sym_max)
    self.scale_lower_bound = LowerBound(scale_bound)
    self.scale_upper_bound = UpperBound(scale_upper_bound) if SCALE_UPPER_BOUND else nn.Identity()
    self.likelihood_lower_bound = LowerBound(ll_bound)

  def sym_bound(self, sym):
    return self.sym_lower_bound(self.sym_upper_bound(sym))

  @staticmethod
  def _standardized_cumulative(x):
    # Using the complementary error function maximizes numerical precision, considering erf(x) == -erf(-x)
    const = float(-(2**-0.5))
    return HALF * torch.erfc(const * x)

  def _likelihood(self, symbols, scales, means):
    # cdf N(u, s) at x == cdf N(0,1) at (x-u)/s
    scales = self.scale_lower_bound(scales)
    scales = self.scale_upper_bound(scales)
    upper = self._standardized_cumulative((symbols - means + HALF) / scales)
    lower = self._standardized_cumulative((symbols - means - HALF) / scales)
    if not (upper >= lower).all():
      print('possible extreme scales', scales.min().detach(), scales.max().detach())
      print('possible outrange y symbols', symbols.min().detach(), symbols.max().detach())
      raise
    return upper - lower
  
  def forward(self, symbols, scales, means):
    # range of symbols must be limited
    assert (symbols >= self.sym_min).all()
    assert (symbols <= self.sym_max).all()
    return self.likelihood_lower_bound(self._likelihood(symbols, scales, means))
  
  def cal_cdf(self, scales, means):
    samples = torch.arange(start=self.sym_min, end=self.sym_max+1, device=scales.device, dtype=scales.dtype)
    # WARNING: extremely huge
    pmf = self._likelihood(
      samples[None, :], 
      scales.detach().flatten()[:, None], 
      means .detach().flatten()[:, None])
    idx = pmf.max(dim=1)[0] <= 0 # find which position idx has the 'mean out of range' problem
    if idx.any(): # mean out of sym range and scale is small
      lim = -(means.flatten()[idx] >= 0).int() # find whether mean is too big or too small
      try:
        pmf[idx, lim] += 1.0
      except:
        print(idx)
        print(lim)
        raise
    return pmf2cdf(pmf)
  
  def compress(self, symbols, scales, means):
    # range of symbols must be limited
    assert (symbols >= self.sym_min).all()
    assert (symbols <= self.sym_max).all()
    sym = symbols.detach().cpu().flatten().to(torch.int32).numpy() - self.sym_min
    cdf = self.cal_cdf(scales, means).cpu().to(torch.int32).numpy()
    ac_enc = yaecl.ac_encoder_t()
    ac_enc.encode_nxn(sym, cdf, 16)
    ac_enc.flush()
    if PRINT_BITS:
      bits_fwd = -torch.log2(self.forward(symbols, scales, means)).sum().item()
      bits_cdf = -np.log2(( cdf[ np.arange(len(sym)), sym+1 ] - cdf[ np.arange(len(sym)), sym ] ) / 2**16).sum()
      bits_coded = ac_enc.bit_stream.size()
      print(f'bits fwd {bits_fwd:.0f}, bits cdf {bits_cdf:.0f}, bits actual {bits_coded}, ratio fwd = {bits_fwd / bits_coded:.4f}, ratio cdf = {bits_cdf / bits_coded:.4f}')
    return ac_enc.bit_stream
  
  def decompress(self, bit_stream, scales, means):
    cdf = self.cal_cdf(scales, means).cpu().to(torch.int32).numpy()
    sym = np.zeros(scales.shape, dtype=np.int32).flatten()
    ac_dec = yaecl.ac_decoder_t(bit_stream)
    ac_dec.decode_nxn(self.sym_max - self.sym_min + 1, memoryview(cdf), 16, memoryview(sym))
    sym = torch.from_numpy(sym + self.sym_min).to(scales.device).to(scales.dtype).reshape(scales.shape)
    return sym