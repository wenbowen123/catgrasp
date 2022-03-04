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
from torch.optim.lr_scheduler import StepLR
import torchvision
from PIL import Image
from transformations import *
from Utils import *



class RotateCloudZAxis:
  def __init__(self,cfg):
    self.cfg = cfg

  def __call__(self,data):
    if np.random.uniform()<self.cfg['rotate_cloud_prob']:
      center = (data['cloud_xyz'].max(axis=0) + data['cloud_xyz'].min(axis=0)) / 2
      tf = np.eye(4)
      tf[:3,3] = -center
      new_tf = np.eye(4)
      z_rot = np.random.uniform(0,np.pi*2)
      new_tf[:3,:3] = euler_matrix(0,0,z_rot,axes='sxyz')[:3,:3]
      tf = new_tf@tf
      new_tf = np.eye(4)
      new_tf[:3,3] = center
      tf = new_tf@tf

      for k in ['cloud_xyz']:
        if k in data:
          data[k] = (tf@to_homo(data[k]).T).T[:,:3]

      for k in ['cloud_normal']:
        if k in data:
          data[k] = (tf[:3,:3]@data[k].T).T

    return data


class FlipCloud:
  def __init__(self,cfg):
    self.cfg = cfg


  def __call__(self,data,axis=['x','y','z']):
    '''
    @axis: flip along axis
    '''
    if np.random.uniform()<self.cfg['flip_cloud_prob']:
      cur_axis = np.random.choice(np.array(axis),size=1)
      dim = ['x','y','z'].index(cur_axis)
      data['cloud_xyz'][:,dim] = -data['cloud_xyz'][:,dim]
      if 'cloud_normal' in data:
        data['cloud_normal'][:,dim] = -data['cloud_normal'][:,dim]
    return data



class NormalizeCloud:
  def __init__(self):
    pass

  def __call__(self,data):
    max_xyz = data['cloud_xyz'].max(axis=0)
    min_xyz = data['cloud_xyz'].min(axis=0)
    scale = (max_xyz-min_xyz).max()
    data['cloud_xyz'] = (data['cloud_xyz']-min_xyz) / (scale+1e-15)
    return data



class DropoutCloud:
  def __init__(self,cfg):
    self.cfg = cfg

  def __call__(self,data):
    if np.random.uniform()<self.cfg['dropout_prob']:
      dropout_ratio = np.random.uniform(0,self.cfg['dropout_max_ratio'])
      n_drop = int(dropout_ratio*len(data['cloud_xyz']))
      drop_ids = np.random.choice(len(data['cloud_xyz']),size=n_drop,replace=False)
      keep_ids = np.array(list(set(np.arange(len(data['cloud_xyz']))) - set(drop_ids)))
      to_replace_ids = np.random.choice(keep_ids,size=n_drop,replace=True)
      for k in ['cloud_xyz','cloud_normal','cloud_rgb','cloud_nocs']:
        if k in data:
          data[k][drop_ids] = data[k][to_replace_ids]

    return data