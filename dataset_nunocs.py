import os.path
import numpy as np
import os,sys,copy,time,cv2,pickle,gzip
from scipy.signal import convolve2d
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from Utils import *
from data_reader import *
from augmentations import *



class NunocsIsolatedDataset(torch.utils.data.Dataset):
  def __init__(self,cfg,phase):
    super().__init__()
    self.cfg = cfg
    self.phase = phase
    self.reader = DataReader(self.cfg)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    if self.phase=='train':
      self.files = sorted(glob.glob(f"{code_dir}/{self.cfg['train_root']}/*.pkl"))
    elif self.phase=='val':
      self.files = sorted(glob.glob(f"{code_dir}/{self.cfg['val_root']}/*.pkl"))
    elif self.phase=='test':
      self.files = []

    print("phase={} #self.files={}".format(self.phase,len(self.files)))


  def __len__(self):
    return len(self.files)


  def transform(self,data):
    keep_ids = np.arange(data['cloud_xyz'].shape[0])
    valid_mask = data['cloud_xyz'][:,2]>=0.1
    keep_ids = keep_ids[valid_mask]
    data['cloud_xyz'] = data['cloud_xyz'][valid_mask]
    replace = data['cloud_xyz'].shape[0]<self.cfg['n_pts']
    ids = np.random.choice(np.arange(data['cloud_xyz'].shape[0]),size=(self.cfg['n_pts']),replace=replace)
    data['cloud_xyz'] = data['cloud_xyz'][ids]
    keep_ids = keep_ids[ids]
    data['cloud_nocs'] = data['cloud_nocs'][keep_ids].reshape(-1,3)/255.0
    data['cloud_rgb'] = data['cloud_rgb'][keep_ids].reshape(-1,3)
    data['cloud_normal'] = data['cloud_normal'][keep_ids].reshape(-1,3)
    data['cloud_xyz_original'] = copy.deepcopy(data['cloud_xyz'])
    data['keep_ids'] = keep_ids

    if self.phase=='train':
      data = DropoutCloud(self.cfg)(data)

    data = NormalizeCloud()(data)
    data['input'] = np.concatenate((data['cloud_xyz'],data['cloud_normal']), axis=-1)

    if 'mean' in self.cfg:
      data['input'] = (data['input']-self.cfg['mean'].reshape(1,-1)) / (self.cfg['std'].reshape(1,-1)+1e-15)

    for k in ['color_file']:
      if k in data:
        del data[k]
    return data



  def __getitem__(self,index):
    file = self.files[index]
    while 1:
      try:
        with gzip.open(file,'rb') as ff:
          data = pickle.load(ff)
        break
      except Exception as e:
        time.sleep(0.001)

    data = self.transform(data)
    return data