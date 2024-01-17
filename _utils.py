import os
import sys
import time
import math
import random
from subprocess import Popen, PIPE
import json

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torchvision import transforms

class GO(object):
  def __init__(self, name=''):
    self.name = name
  def __enter__(self):
    pass
  def __exit__(self, type, value, traceback):
    pass

class AverageMeter(object):
  '''Compute running average.'''
  def __init__(self):
    self.val = 0
    self.sum = 0
    self.cnt = 0
    self.avg = 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def find_type(x, ttype=bytes):
  if type(x) == list:
    return [b for _x in x          for b in find_type(_x, ttype=ttype)]
  elif type(x) == dict:
    return [b for _x in x.values() for b in find_type(_x, ttype=ttype)]
  elif type(x) == ttype:
    return [x]
  else:
    return []

def rupdate(d, u):
  for k, v in u.items():
    if isinstance(v, dict):
      d[k] = rupdate(d.get(k, {}), v)
    else:
      d[k] = v
  return d

def check_cfg_allset(cfg, k_prefix=''):
  flag = True
  for k, v in cfg.items():
    if isinstance(v, dict):
      flag &= check_cfg_allset(v, k)
    elif v is None:
      print(f'\'{k_prefix}.{k}\' is not set')
      flag = False
  return flag

IMG_EXTENSIONS = (
  'jpg',
  'jpeg',
  'png',
  'ppm',
  'bmp',
  'pgm',
  'tif',
  'tiff',
  'webp',
)
def collect_images(rootdir, recursive=False):
  assert not recursive, 'recursive not longer supported'
  all_img = filter( lambda s:s.split('.')[-1].lower() in IMG_EXTENSIONS, sorted(os.listdir(rootdir)) )
  return [os.path.realpath(os.path.join(rootdir, img)) for img in all_img]

def read_img_np(img_fpath):
  img = Image.open(img_fpath).convert('RGB')
  return np.array(img, dtype=np.float32) / 255.0

def read_img_pt(img_fpath):
  img = Image.open(img_fpath).convert('RGB')
  return transforms.ToTensor()(img)

def torch_init(seed=3407):
  torch.backends.cudnn.benchmark = False     # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest. 
  torch.backends.cudnn.deterministic = True  # A bool that, if True, causes cuDNN to only use deterministic convolution algorithms.

  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)