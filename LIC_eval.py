import os
import time
import argparse
from math import log

import sqlite3 
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ms_ssim

from yaecl import bit_stream_t
from compressai_dep import rename_pretrained

from _utils import find_type, collect_images, read_img_pt, torch_init
import models

def l2bit(l):
  return -(torch.log(l) / log(2)).sum()  

def eval_model_on_image(img_fpath, model):
  x = read_img_pt(img_fpath).cuda().unsqueeze(0)
  x_rec = x.clone()

  for rc in range(50):
    out_enc = model.compress(x_rec.detach())
    out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
    x_rec = out_dec['x_hat']
    mse_rc = ((x_rec - x)**2).mean().item()
    ssim_rc = ms_ssim(x_rec, x, data_range=1.0).item()
    print(f're-compression={rc+1}, mse={mse_rc:.2e}, ssim={ssim_rc:.4f}', flush=True)

  bpp = 0
  for k, s in out_enc['strings'].items():
    s_bytes        = find_type(s, ttype=bytes)
    s_bit_stream_t = find_type(s, ttype=bit_stream_t)
    assert (s_bytes == []) != (s_bit_stream_t == [])
    bpp_k = 0
    bpp_k += sum(len(_s)*8.0 for _s in s_bytes)
    bpp_k += sum(_s.size()   for _s in s_bit_stream_t)
    bpp_k /= x.size(2) * x.size(3)
    bpp += bpp_k

  return bpp, mse_rc, ssim_rc

def model_from_checkpoint(modelclass, checkpoint):
  state_dict = rename_pretrained(checkpoint['net'])
  net = modelclass(**checkpoint['net_def'])
  net.load_state_dict(state_dict)
  return net
  
def main():
  args = get_args()

  model = model_from_checkpoint(eval(f'models.{args.modelclass}'), torch.load(args.path)).eval().cuda()
  model.update()

  bpp_list  = []
  psnr_list = []
  ssim_list = []

  for img_fpath in (pbar := tqdm(collect_images(os.path.realpath(args.rootdir)))):
    pbar.set_description(img_fpath)
    bpp, mse, ssim = eval_model_on_image(img_fpath, model)
    bpp_list  += [bpp]
    psnr_list += [-10 * np.log10(mse)]
    ssim_list += [ssim]

  print(f'bpp = {np.mean(bpp_list):8.4f}, psnr = {np.mean(psnr_list):8.4f}, ssim = {np.mean(ssim_list):8.4f}')

def get_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-d', '--rootdir',    help='image dataset root dir', required=True)
  parser.add_argument('-c', '--modelclass', help='model class',            required=True)
  parser.add_argument('-p', '--path',       help='checkpoint path',        required=True)

  args = parser.parse_args()  
  return args

if __name__ == '__main__':
  torch_init()
  torch.set_grad_enabled(False)
  torch.backends.cudnn.deterministic = True  # A bool that, if True, causes cuDNN to only use deterministic convolution algorithms.

  main()