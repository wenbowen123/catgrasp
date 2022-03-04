import os.path
import numpy as np
import os,sys,copy,time,cv2
from scipy.signal import convolve2d
code_dir = os.path.dirname(os.path.realpath(__file__))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from Utils import *


class NocsMinSymmetryCELoss(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.cfg = cfg
    self.symmetry_tfs = get_symmetry_tfs(self.cfg['nocs_class_name'])
    new_tfs = []
    for symmetry_tf in self.symmetry_tfs:
      tf = torch.from_numpy(symmetry_tf).cuda().float()
      new_tfs.append(tf)
    self.symmetry_tfs = torch.stack(new_tfs, dim=0)
    self.n_sym = len(self.symmetry_tfs)
    self.bin_resolution = 1/self.cfg['ce_loss_bins']

  def forward(self,pred,target):
    B,N = target.shape[:2]
    tmp_target = torch.matmul(self.symmetry_tfs.unsqueeze(0).expand(B,self.n_sym,4,4), to_homo_torch(target-0.5).permute(0,2,1).unsqueeze(1).expand(B,self.n_sym,4,-1))
    tmp_target = tmp_target.permute(0,1,3,2)[...,:3] + 0.5
    cloud_nocs_bin_class = torch.clamp(tmp_target/self.bin_resolution,0,self.cfg['ce_loss_bins']-1).long()

    pred = pred.reshape(B,-1,3,self.cfg['ce_loss_bins']).unsqueeze(-1).expand(-1,-1,-1,-1,self.n_sym)

    loss = []
    for i in range(3):
      loss.append(nn.CrossEntropyLoss(reduction='none')(pred[:,:,i].permute(0,2,3,1), cloud_nocs_bin_class[...,i]))
    loss = torch.stack(loss,dim=-1).sum(dim=-1)
    loss = loss.mean(dim=-1)
    ids = loss.argmin(dim=1)
    loss = torch.gather(loss,dim=1,index=ids.unsqueeze(1))
    loss = loss.mean()
    return loss
