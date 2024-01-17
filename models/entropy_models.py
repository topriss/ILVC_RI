# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS `AS IS` AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compressai_dep import available_entropy_coders, default_entropy_coder, pmf_to_quantized_cdf, LowerBound, ans

HALF = float(0.5)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class _EntropyCoder:
  def __init__(self, method):
    assert isinstance(method, str), f'Invalid method type `{type(method)}`'
    assert method in available_entropy_coders(), f'Unknown entropy coder `{method}`, available: {", ".join(available_entropy_coders())})'
    if method == 'ans':
      encoder = ans.RansEncoder()
      decoder = ans.RansDecoder()
    elif method == 'rangecoder':
      import range_coder
      encoder = range_coder.RangeEncoder()
      decoder = range_coder.RangeDecoder()
    else:
      raise NotImplementedError
    self.name = method
    self._encoder = encoder
    self._decoder = decoder

  def encode_with_indexes(self, *args, **kwargs):
    return self._encoder.encode_with_indexes(*args, **kwargs)

  def decode_with_indexes(self, *args, **kwargs):
    return self._decoder.decode_with_indexes(*args, **kwargs)

class EntropyModel(nn.Module):
  def __init__(self, entropy_coder=None, entropy_coder_precision=16, likelihood_bound=1e-9):
    super().__init__()
    self.entropy_coder = _EntropyCoder(default_entropy_coder() if entropy_coder is None else entropy_coder)
    self.entropy_coder_precision = int(entropy_coder_precision)
    self.likelihood_lower_bound = LowerBound(likelihood_bound) if likelihood_bound > 0 else nn.Identity()
    # to be filled on update()
    self.register_buffer('_quantized_cdf', torch.IntTensor())
    self.register_buffer('_cdf_length', torch.IntTensor())
    self.register_buffer('_offset', torch.IntTensor())

  def forward(self, *args: Any) -> Any:
    raise NotImplementedError()

  def _pmf_to_cdf(self, pmf, pmf_length, max_length):
    cdf = torch.zeros([len(pmf_length), max_length + 2], dtype=torch.int32, device=pmf.device)
    for i, p in enumerate(pmf):
      tail_mass = torch.maximum(
        1 - p[: pmf_length[i]].sum(dim=0, keepdim=True), 
        p.new_tensor(0) )
      prob = torch.cat((p[: pmf_length[i]], tail_mass), dim=0)
      _cdf = torch.IntTensor(pmf_to_quantized_cdf(prob.tolist(), self.entropy_coder_precision))
      cdf[i, : _cdf.size(0)] = _cdf
    return cdf

  def _check_cdf_size(self):
    assert self._quantized_cdf.numel() > 0, 'Uninitialized CDFs. Run update() first'
    assert self._quantized_cdf.size().__len__() == 2, f'Invalid CDF size {self._quantized_cdf.size()}'

  def _check_cdf_length(self):
    assert self._cdf_length.numel() > 0, 'Uninitialized CDF lengths. Run update() first'
    assert self._cdf_length.size().__len__() == 1, f'Invalid offsets size {self._cdf_length.size()}'

  def _check_offsets_size(self):
    assert self._offset.numel() > 0, 'Uninitialized offsets. Run update() first'
    assert self._offset.size().__len__() == 1, f'Invalid offsets size {self._offset.size()}'

  def compress(self, symbols, indexes, cdf_sancheck=False):
    '''
    Compress quantized tensors to char strings.

    Args:
      symbols (torch.Tensor): quantized tensors
      indexes (torch.IntTensor): tensors CDF indexes
    '''
    assert symbols.size().__len__() >= 2, 'Invalid `symbols` size. Expected a tensor with at least 2 dimensions (batch first).'
    assert symbols.size() == indexes.size(), '`symbols` and `indexes` should have the same size.'
    self._check_cdf_size()
    self._check_cdf_length()
    self._check_offsets_size()

    strings = []
    for i in range(symbols.size(0)): # batch dim
      rv = self.entropy_coder.encode_with_indexes(
        symbols[i].reshape(-1).int().tolist(),
        indexes[i].reshape(-1).int().tolist(),
        self._quantized_cdf.tolist(),
        self._cdf_length.reshape(-1).int().tolist(),
        self._offset.reshape(-1).int().tolist(),
      ) # type: bytes
      strings.append(rv)
    if not cdf_sancheck:
      return strings
    else:
      precision = 16.0
      syms = symbols.reshape(-1).int()
      idxs = indexes.reshape(-1).int()
      syms -= self._offset.reshape(-1).int()[idxs.tolist()]
      syms = torch.maximum(syms, torch.zeros_like(syms))
      syms = torch.minimum(syms, self._cdf_length.reshape(-1).int()[idxs.tolist()]-2)
      pmfs = self._quantized_cdf.int()[idxs.tolist(), (syms+1).tolist()] \
           - self._quantized_cdf.int()[idxs.tolist(),  syms   .tolist()]
      bits = -torch.log2(pmfs.float()) + precision
      return strings, bits

  def decompress(self, strings, indexes, dtype=torch.float32):
    '''
    Decompress char strings to quantized tensors.

    Args:
      strings (str): compressed char strings
      indexes (torch.IntTensor): tensors CDF indexes
    '''
    assert isinstance(strings, (tuple, list))
    assert len(strings) == indexes.size(0), f'`strings` and `indexes` should have the same first dim (batch)'
    assert indexes.size().__len__() >= 2, f'Invalid `indexes` size. Expected a tensor with at least 2 dimensions.'
    self._check_cdf_size()
    self._check_cdf_length()
    self._check_offsets_size()

    symbols = torch.empty_like(indexes, dtype=dtype)
    for i in range(len(strings)): # batch dim
      rv = self.entropy_coder.decode_with_indexes(
        strings[i],
        indexes[i].reshape(-1).int().tolist(),
        self._quantized_cdf.tolist(),
        self._cdf_length.reshape(-1).int().tolist(),
        self._offset.reshape(-1).int().tolist(),
      ) # type: list
      symbols[i] = symbols.new_tensor(rv).reshape(symbols[i].size())
    return symbols

class FullyFactorized(EntropyModel):
  r'''
  Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
  S. J. Hwang, N. Johnston, in `Variational image compression with a scale
  hyperprior <https://arxiv.org/abs/1802.01436>`_.

  This is a re-implementation of the entropy bottleneck layer in
  *tensorflow/compression*. See the original paper and the `tensorflow
  documentation
  <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
  for an introduction.
  '''
  def __init__(
    self,
    channels: int,
    *args: Any,
    filters: Tuple[int, ...] = (3, 3, 3, 3),
    init_scale: float = 10,
    tail_mass: float = 1e-9,
    **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)
    self.filters = tuple(int(f) for f in filters)
    all_filters = (1,) + self.filters + (1,)
    scale = init_scale ** (1 / (len(self.filters) + 1))

    for i in range(len(self.filters) + 1):
      matrix = Tensor(channels, all_filters[i + 1], all_filters[i])
      bias   = Tensor(channels, all_filters[i + 1], 1)
      matrix.data.fill_( np.log(np.expm1(1 / scale / all_filters[i + 1])) )
      nn.init.uniform_(bias, -0.5, 0.5)
      self.register_parameter(f'_matrix{i:d}', nn.Parameter(matrix))
      self.register_parameter(f'_bias{i:d}', nn.Parameter(bias))
      if i < len(self.filters):
        factor = Tensor(channels, all_filters[i + 1], 1)
        nn.init.zeros_(factor)
        self.register_parameter(f'_factor{i:d}', nn.Parameter(factor))

    self.quantiles = nn.Parameter(Tensor([-init_scale, 0, init_scale]).repeat(channels, 1, 1))
    target = np.log(2 / tail_mass - 1)
    self.register_buffer('target', Tensor([-target, 0, target]))

  def _logits_cumulative(self, symbols: Tensor, fix_filters: bool=False) -> Tensor:
    # TorchScript not yet working (nn.Module indexing not supported)
    logits = symbols
    for i in range(len(self.filters) + 1):
      matrix = getattr(self, f'_matrix{i:d}')
      bias   = getattr(self, f'_bias{i:d}')
      if fix_filters:
        matrix = matrix.detach()
        bias   = bias  .detach()
      logits = torch.matmul(F.softplus(matrix), logits) + bias
      if i < len(self.filters):
        factor = getattr(self, f'_factor{i:d}')
        if fix_filters:
          factor = factor.detach()
        logits += torch.tanh(factor) * torch.tanh(logits)
    return logits

  def loss(self) -> Tensor:
    logits = self._logits_cumulative(self.quantiles, fix_filters=True)
    loss = torch.abs(logits - self.target).sum()
    return loss

  @torch.jit.unused
  def _likelihood(self, symbols: Tensor) -> Tensor:
    lower = self._logits_cumulative(symbols - HALF)
    upper = self._logits_cumulative(symbols + HALF)
    if not (upper >= lower).all():
      print('possible outrange z symbols', symbols.min().detach(), symbols.max().detach())
      raise
    return torch.sigmoid(upper) - torch.sigmoid(lower)

  def forward(self, symbols: Tensor) -> Tensor:
    assert not torch.jit.is_scripting(), 'TorchScript not yet supported'
    # from B x C x ... to C x B x ...
    perm = np.arange(len(symbols.shape))
    perm[0], perm[1] = perm[1], perm[0]
    symbols = symbols.permute(*perm).contiguous()
    shape = symbols.size()
    symbols = symbols.reshape(shape[0], 1, -1)
    likelihood = self._likelihood(symbols)
    likelihood = self.likelihood_lower_bound(likelihood)
    return likelihood.reshape(shape).permute(*perm).contiguous()

  def update(self, force: bool = False) -> bool:
    if self._offset.numel() > 0 and not force:
      return False

    minima = torch.floor(self.quantiles[:, 0, 0]).int() # int
    maxima = torch.ceil (self.quantiles[:, 0, 2]).int() # int
    pmf_length = maxima - minima + 1
    max_length = pmf_length.max().item()
    samples = torch.arange(max_length, device=self.quantiles.device)
    samples = samples[None, :] + minima[:, None, None]

    pmf = self._likelihood(samples)[:, 0, :].detach()

    self._quantized_cdf = self._pmf_to_cdf(pmf, pmf_length, max_length)
    self._cdf_length = pmf_length + 2
    self._offset = minima

    return True

  def build_indexes(self, size): # size = [B, C, ...]
    view_dims = [-1 if i == 1 else 1 for i in range(len(size))]
    return torch.arange(size[1]).int().view(*view_dims).expand(*size).to(self.quantiles.device)

class GaussianConditional(EntropyModel):
  r'''
  Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
  S. J. Hwang, N. Johnston, in `Variational image compression with a scale
  hyperprior <https://arxiv.org/abs/1802.01436>`_.

  This is a re-implementation of the Gaussian conditional layer in
  *tensorflow/compression*. See the `tensorflow documentation
  <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
  for more information.
  '''

  def __init__(
    self,
    scale_table: Optional[Union[List, Tuple]],
    *args: Any,
    scale_bound: float = 0.11,
    tail_mass: float = 1e-9,
    **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)
    if scale_table:
      assert len(scale_table) > 0
      assert all(s > 0 for s in scale_table)
      assert scale_table == sorted(scale_table)
      scale_bound = min(scale_bound, scale_table[0])
    assert scale_bound > 0
    self.scale_lower_bound = LowerBound(scale_bound)
    self.register_buffer( 'scale_table', self._prepare_scale_table(scale_table) if scale_table else Tensor() )
    self.register_buffer( 'scale_bound', Tensor([scale_bound]))
    self.tail_mass = float(tail_mass)

  @staticmethod
  def _prepare_scale_table(scale_table):
    return Tensor(tuple(float(s) for s in scale_table))

  @staticmethod
  def _standardized_cumulative(x: Tensor) -> Tensor:
    # Using the complementary error function maximizes numerical precision, considering erf(x) == -erf(-x)
    const = float(-(2**-0.5))
    return HALF * torch.erfc(const * x)

  @staticmethod
  def _standardized_quantile(c):
    return scipy.stats.norm.ppf(c) # inverse of cdf

  def _likelihood(self, symbols: Tensor, scales: Tensor) -> Tensor:
    # cdf N(u, s) at x == cdf N(0,1) at (x-u)/s
    scales = self.scale_lower_bound(scales)
    upper = self._standardized_cumulative((symbols + HALF) / scales)
    lower = self._standardized_cumulative((symbols - HALF) / scales)
    assert (upper >= lower).all()
    return upper - lower

  def forward(self, symbols: Tensor, scales: Tensor) -> Tensor:
    likelihood = self._likelihood(symbols, scales)
    likelihood = self.likelihood_lower_bound(likelihood)
    return likelihood

  def update(self, scale_table=None, force=False):
    if self._offset.numel() > 0 and not force:
      return False

    scale_table = get_scale_table() if scale_table is None else scale_table
    self.scale_table = self._prepare_scale_table(scale_table).to(self.scale_table.device)

    multiplier = -self._standardized_quantile(self.tail_mass / 2)
    pmf_center = torch.ceil(self.scale_table * multiplier).int()
    pmf_length = 2 * pmf_center + 1
    max_length = torch.max(pmf_length).item()
    samples = torch.arange(max_length, device=pmf_center.device).float() - pmf_center[:, None] # shape == [len(scalt_table), max_length]
    samples_scale = self.scale_table.unsqueeze(1).float()

    pmf = self._likelihood(samples, samples_scale).detach()

    self._quantized_cdf = self._pmf_to_cdf(pmf, pmf_length, max_length)
    self._cdf_length = pmf_length + 2
    self._offset = -pmf_center

    return True

  def build_indexes(self, scales: Tensor) -> Tensor:
    scales = self.scale_lower_bound(scales)
    indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
    for s in self.scale_table[:-1]:
      indexes -= (scales <= s).int()
    return indexes