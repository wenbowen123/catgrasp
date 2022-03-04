import os.path
import numpy as np
import os,sys,copy,time,cv2,pickle,gzip
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
from Utils import *


class DataReader:
  def __init__(self,cfg):
    self.cfg = cfg

  def read_data_by_colorfile(self,color_file,fetch=['nocs_map','xyz_map','normal_map']):
    rgb = np.array(Image.open(color_file))
    depth = cv2.imread(color_file.replace('rgb','depth'),-1)/1e4
    depth[depth<0.1] = 0
    depth[depth>self.cfg['zfar']] = 0
    seg = cv2.imread(color_file.replace('rgb','seg'),-1).astype(int)
    with open(color_file.replace('rgb.png','meta.pkl'),'rb') as ff:
      meta = pickle.load(ff)
      K = meta['K']
      env_body_ids = meta['env_body_ids']

    data = {'rgb':rgb,'seg':seg, 'depth':depth, 'color_file':color_file}

    if 'nocs_map' in fetch:
      data['nocs_map'] = np.array(Image.open(color_file.replace('rgb','nunocs')))


    if 'xyz_map' in fetch:
      data['xyz_map'] = depth2xyzmap(depth,K)

    if 'normal_map' in fetch:
      data['normal_map'] = read_normal_image(color_file.replace('rgb','normal'))

    for id in env_body_ids:
      mask = seg==id
      for k in ['rgb','depth','seg','nocs_map','xyz_map','normal_map']:
        if k in data:
          data[k][mask] = 0


    return data


