# Entropy models + quantization, just for convenience

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai_dep import update_registered_buffers

from .utils import ccat
from .entropy_models import FullyFactorized, GaussianConditional

class CompressionBasic(nn.Module):
  def __init__(self):
    super().__init__()
  
  def aux_loss(self):
    return sum( m.loss() for m in self.modules() if isinstance(m, FullyFactorized) )
  
  def update(self, force=False):
    updated = False
    for m in self.modules():
      if isinstance(m, (FullyFactorized, GaussianConditional)): 
        updated |= m.update(force=force)
    return updated

  def load_state_dict(self, state_dict):
    for n, m in self.named_modules():
      if isinstance(m, FullyFactorized):
        update_registered_buffers(m, n, ['_quantized_cdf', '_offset', '_cdf_length'],                state_dict)
      elif isinstance(m, GaussianConditional):
        update_registered_buffers(m, n, ['_quantized_cdf', '_offset', '_cdf_length', 'scale_table'], state_dict)
    super().load_state_dict(state_dict)