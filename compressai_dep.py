###################################################################################################################################

from compressai.models.utils import update_registered_buffers
from compressai.zoo.pretrained import load_pretrained as rename_pretrained

from compressai.transforms.functional import rgb2ycbcr, ycbcr2rgb
from compressai.ops import LowerBound
from compressai import ans
from compressai import available_entropy_coders
from compressai import get_entropy_coder as default_entropy_coder
from compressai._CXX import pmf_to_quantized_cdf
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from compressai.models.utils import conv, deconv
from compressai.layers import conv1x1, conv3x3, subpel_conv3x3, GDN, ResidualBlock, MaskedConv2d

###################################################################################################################################

from pathlib import Path
from random import randrange, random

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageFolder(Dataset):
  """
  - rootdir/
    - train/
      - img000.png
      - img001.png
    - test/
      - img000.png
      - img001.png
  """
  def __init__(self, root, split, transform=None, random_sample=0, lossless=False, noise=False):
    splitdir = Path(root) / split
    if not splitdir.is_dir():
      raise RuntimeError(f'Invalid directory "{root}"')
    self.samples = [f for f in splitdir.iterdir() if f.is_file()]
    self.transform = transform

    self.random_sample = random_sample
    if random_sample > 0:
      print(f'using random sampling with len={random_sample}')
    self.lossless = lossless
    if lossless:
      print('using Lanczos rescale to remove jpeg artifect')
    self.noise = noise
    if noise:
      print('generating noisy-clean pairs with synthetic noise')

  def __getitem__(self, index):
    if self.random_sample > 0:
      index = randrange(len(self.samples))
    while True:
      try:
        img = Image.open(self.samples[index]).convert("RGB")
        break
      except:
        index = (index + 1) % self.samples.__len__()
    if self.lossless: # preprocessing on openimage dataset to remove jpeg artifact, see https://openaccess.thecvf.com/content_CVPR_2020/papers/Mentzer_Learning_Better_Lossless_Compression_Using_Lossy_Compression_CVPR_2020_paper.pdf
      scale = 0.6 + 0.2 * random()
      (width, height) = ( int(img.width * scale), int(img.height * scale) )
      img = img.resize((width, height), resample=Image.Resampling.LANCZOS)

    img = self.transform(img)

    return img

  def __len__(self):
    return self.random_sample if self.random_sample > 0 else len(self.samples)

def configure_optimizers(net, lr, lr_aux):
  '''
  Separate parameters for the main optimizer and the auxiliary optimizer. Return two optimizers
  '''
  parameters     = { n for n, p in net.named_parameters() if not ( n.endswith('.quantiles') or n.endswith('.quantiles.original') ) }
  aux_parameters = { n for n, p in net.named_parameters() if     ( n.endswith('.quantiles') or n.endswith('.quantiles.original') ) }

  # Make sure no repetition and no omission
  params_dict = { n:p for n, p in net.named_parameters() }
  inter_params = parameters & aux_parameters
  union_params = parameters | aux_parameters
  assert len(inter_params) == 0
  assert len(union_params) == len(params_dict.keys()) 

  optimizer     = Adam([params_dict[n] for n in parameters    ], lr=lr)
  aux_optimizer = Adam([params_dict[n] for n in aux_parameters], lr=lr_aux)
  return optimizer, aux_optimizer

class UpperBound(nn.Module):
  def __init__(self, bound):
    super().__init__()
    self.lower_bound = LowerBound(-bound)
  def forward(self, x):
    return -self.lower_bound(-x)

class ResidualBlockWithStride(nn.Module):

  def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
    super().__init__()
    self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
    self.leaky_relu = nn.LeakyReLU(inplace=False)
    self.conv2 = conv3x3(out_ch, out_ch)
    self.skip = conv1x1(in_ch, out_ch, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.conv1(x)
    out = self.leaky_relu(out)
    out = self.conv2(out)
    out = self.leaky_relu(out)
    identity = self.skip(x)
    out += identity
    return out

class ResidualBlockUpsample(nn.Module):

  def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
    super().__init__()
    self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
    self.leaky_relu = nn.LeakyReLU(inplace=False)
    self.conv = conv3x3(out_ch, out_ch)
    self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.subpel_conv(x)
    out = self.leaky_relu(out)
    out = self.conv(out)
    out = self.leaky_relu(out)
    identity = self.upsample(x)
    out += identity
    return out

###################################################################################################################################