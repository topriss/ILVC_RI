import os
import sys
import shutil
from math import log, log2
import json
from collections import defaultdict
from importlib.machinery import SourceFileLoader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.parametrize as P
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from compressai_dep import ImageFolder, configure_optimizers
from pytorch_msssim import ms_ssim

from _utils import GO, torch_init, AverageMeter, rupdate, check_cfg_allset
from models import *

def l2bit(l):
  return -(torch.log(l) / log(2)).sum()

class RateDistortionLoss(nn.Module):
  def __init__(self, net, lbd):
    super().__init__()
    self.net = net
    self.lbd = lbd

  def forward(self, img):
    with P.cached():
      ret = self.net(img)
    N, _, H, W = img.size()
    N_batch_pixel = N * H * W   
    loss = {}  
    loss['mse']  = nn.MSELoss()(ret['x_hat'], img)
    for k, ll in ret['likelihoods'].items():
      loss[f'bpp.{k}'] = l2bit(ll) / N_batch_pixel
      loss['bpp'] = loss.get('bpp', 0.0) + loss[f'bpp.{k}']
    loss['loss'] = loss['bpp'] + self.lbd * 255**2 * loss['mse']
    loss['aux_loss'] = self.net.aux_loss()
    return loss

EPOCH = 0

def train_epoch(train_dataloader, netNloss, optimizer, aux_optimizer, writer, clip=1.0):
  netNloss.train()

  for i, img in enumerate(tqdm(train_dataloader, dynamic_ncols=True, mininterval=1)):
    loss = netNloss(img.cuda())
    optimizer    .zero_grad()
    aux_optimizer.zero_grad()
    loss['loss']    .backward()
    loss['aux_loss'].backward()
    if clip > 0:
      torch.nn.utils.clip_grad_norm_(netNloss.parameters(), clip)
    optimizer    .step()
    aux_optimizer.step()

def valid_epoch(valid_dataloader, netNloss, writer):
  netNloss.eval()

  aloss = defaultdict(AverageMeter)
  for i, img in enumerate(tqdm(valid_dataloader, dynamic_ncols=True, mininterval=1)):
    with torch.no_grad():
      loss = netNloss(img.cuda())
    for k, v in loss.items():
      writer.add_scalar(f'valid/{k}', v.item(), EPOCH*len(valid_dataloader)+i, new_style=True)
      aloss[k].update(v.item())
  aloss = {k: v.avg for k, v in aloss.items()}
  print(json.dumps(aloss, indent=2, sort_keys=True))

  return aloss['loss']

def main():
  torch_init()

  cfg = get_cfg()

  with GO('model'):
    net = eval(cfg['net_class'])(**cfg['net'])
    netNloss = RateDistortionLoss(net, cfg['lbd']).cuda()
    print(net)

  with GO('optimization'):
    optimizer, aux_optimizer = configure_optimizers(netNloss.net, cfg['lr'], cfg['lr_aux'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, threshold=0.1, verbose=True)

  with GO('data'):
    train_transforms = T.Compose([ T.RandomCrop(cfg['patchsize']), T.ToTensor() ])
    valid_transforms = T.Compose([ T.CenterCrop(cfg['patchsize']), T.ToTensor() ])
    train_dataset = ImageFolder(cfg['dataset'], split='train', transform=train_transforms)
    valid_dataset = ImageFolder(cfg['dataset'], split='valid', transform=valid_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train_batchsize'], num_workers=64, shuffle=True,  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['valid_batchsize'], num_workers=64, shuffle=False, pin_memory=True)
  
  with GO('main loop'):
    global EPOCH
    writer = SummaryWriter(cfg['workdir'])
    best_loss = float('inf')
    for epoch in range(cfg['epochs']):
      print(f'epoch {epoch}, lr={optimizer.param_groups[0]["lr"]:.1e}')
      EPOCH = epoch
      _    = train_epoch(train_dataloader, netNloss, optimizer, aux_optimizer, writer)
      loss = valid_epoch(valid_dataloader, netNloss,                           writer)
      lr_scheduler.step(loss)

      is_best = loss < best_loss
      best_loss = min(loss, best_loss)
      state = {
        'net_def':       cfg['net'],
        'net':           netNloss.net .state_dict(),
      }
      checkpoint_path = f'{cfg["workdir"]}/checkpoint.pt'
      torch.save(state, checkpoint_path)
      if is_best:
        shutil.copyfile(checkpoint_path, f'{cfg["workdir"]}/checkpoint_best.pt')
    writer.close()

def get_cfg():
  default_cfg = {
    'net': {
    },
    'q': None, # quality, 1-8

    'lr':     1e-4,
    'lr_aux': 1e-3,
    'epochs': 80,

    'dataset': None, 
    'train_batchsize': 16,
    'valid_batchsize': 64,
    'patchsize': (256, 256),

    'workdir': './workdir'
  }

  q_lbd = {
    1: 0.0018, 
    2: 0.0035, 
    3: 0.0067, 
    4: 0.0130, 
    5: 0.0250, 
    6: 0.0483, 
    7: 0.0932, 
    8: 0.1800,
  }

  cfg_fpath = sys.argv[1]
  cfg = rupdate(default_cfg, SourceFileLoader('', cfg_fpath).load_module().cfg)
  cfg['net_class'] = os.path.basename(cfg_fpath).split('.')[0]
  assert check_cfg_allset(cfg)
  
  cfg['lbd'] = q_lbd[cfg['q']]
  cfg['dataset'] = os.path.realpath(cfg['dataset'])
  cfg['workdir'] += '/' + os.path.basename(cfg_fpath).rsplit('.', 1)[0]
  os.system(f'mkdir -p {cfg["workdir"]}')
  os.system(f'cp {__file__} {cfg["workdir"]}/')
  
  print(json.dumps(cfg, indent=2))
  return cfg

if __name__=='__main__':
  main()