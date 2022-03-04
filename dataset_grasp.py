import os.path
import re,yaml,os,sys
import numpy as np
from collections import defaultdict
import os,sys,copy,time,cv2,pickle,gzip
from scipy.signal import convolve2d
code_dir = os.path.dirname(os.path.realpath(__file__))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dexnet.grasping.gripper import RobotGripper
from PIL import Image
from Utils import *
from data_reader import *
from augmentations import *
import matplotlib.pyplot as plt


class GraspDataset(torch.utils.data.Dataset):
  def __init__(self,cfg,phase,class_name=None):
    super().__init__()
    self.cfg = cfg
    self.phase = phase
    if class_name is not None:
      self.class_name = class_name
    else:
      self.class_name = cfg['class_name']
    self.symmetry_tfs = get_symmetry_tfs(self.class_name)
    self.symmetry_tfs = np.array(self.symmetry_tfs)
    if self.phase=='train':
      self.files = sorted(glob.glob(f"{code_dir}/{self.cfg['train_root']}/*grasp.pkl"))
    elif self.phase=='val':
      self.files = sorted(glob.glob(f"{code_dir}/{self.cfg['val_root']}/*grasp.pkl"))
    elif self.phase=='test':
      self.files = []
    else:
      raise RuntimeError

    self.keys = []
    for file in self.files:
      color_file = file.replace('grasp.pkl','rgb.png')
      with gzip.open(file,'rb') as ff:
        grasps = pickle.load(ff)
        for body_id,vv in grasps.items():
          for i_grasp in range(len(vv)):
            grasp_in_cam,score = vv[i_grasp]
            self.keys.append((color_file,body_id,grasp_in_cam,score))

    ids = np.random.choice(len(self.keys),size=min(200000,len(self.keys)),replace=False)
    self.keys = np.array(self.keys)[ids]

    self.classes = np.array(self.cfg['classes'])
    print("phase={} #self.keys={}".format(self.phase,len(self.keys)))



  def __len__(self):
    return len(self.keys)


  def transform(self,data,grasp_pose):
    valid_mask = data['cloud_xyz'][:,2]>=0.1
    data['cloud_xyz'] = data['cloud_xyz'][valid_mask].reshape(-1,3)
    data['cloud_normal'] = data['cloud_normal'][valid_mask].reshape(-1,3)

    ############ Transform to grasp frame
    data['cloud_xyz'] = (np.linalg.inv(grasp_pose)@to_homo(data['cloud_xyz']).T).T[:,:3]
    data['cloud_normal'] = (np.linalg.inv(grasp_pose[:3,:3])@data['cloud_normal'].T).T

    replace = data['cloud_xyz'].shape[0]<self.cfg['n_pts']
    ids = np.random.choice(np.arange(data['cloud_xyz'].shape[0]),size=(self.cfg['n_pts']),replace=replace)
    data['cloud_xyz'] = data['cloud_xyz'][ids]
    data['cloud_normal'] = data['cloud_normal'][ids].reshape(-1,3)

    ############# Augmentations
    if self.phase=='train':
      data = FlipCloud(self.cfg)(data,axis=['y'])

    data['cloud_xyz_original'] = copy.deepcopy(data['cloud_xyz'])
    data['input'] = np.concatenate((data['cloud_xyz'],data['cloud_normal']), axis=-1)

    if 'mean' in self.cfg:
      data['input'] = (data['input']-self.cfg['mean'].reshape(1,-1)) / (self.cfg['std'].reshape(1,-1)+1e-15)

    for k in ['color_file','cloud_rgb','cloud_nocs']:
      if k in data:
        del data[k]

    return data



  def __getitem__(self,index):
    color_file,body_id,grasp_pose,score = self.keys[index]
    with gzip.open(color_file.replace('/train/','/train_isolated_nunocs/').replace('/test/','/test_isolated_nunocs/').replace('rgb.png','_seg{}.pkl'.format(body_id)),'rb') as ff:
      data = pickle.load(ff)

    data['score'] = np.digitize(score, self.classes)-1
    data = self.transform(data,grasp_pose)

    return data