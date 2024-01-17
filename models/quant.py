# state-less quntization functions 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def aun(x):
  return x + torch.empty_like(x).uniform_(-0.5, 0.5) 

def ste(x):
  return (torch.round(x) - x).detach() + x