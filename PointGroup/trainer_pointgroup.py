import os.path
import numpy as np
import os,sys,copy,time,cv2,tqdm
from scipy.signal import convolve2d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append('{}/../'.format(code_dir))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from PIL import Image
from model.pointgroup.pointgroup import PointGroup
from model.pointgroup.pointgroup import model_fn_decorator
from data.dataset_seg import Dataset
from util.config import get_parser
from Utils import *

class TrainerPointGroup:
  def __init__(self,cfg):
    self.cfg = cfg
    self.epoch = 0

    self.best_train = 1e9
    self.best_val = 1e9

    self.train_data = Dataset(self.cfg,phase='train')
    self.val_data = Dataset(self.cfg,phase='val')

    self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.cfg['n_workers'], pin_memory=False, drop_last=True,worker_init_fn=worker_init_fn,collate_fn=self.train_data.merge)
    self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.cfg['n_workers'], pin_memory=False, drop_last=False,worker_init_fn=worker_init_fn,collate_fn=self.val_data.merge)

    self.cfg_pg = get_parser()
    self.model = PointGroup(self.cfg_pg)
    self.model = nn.DataParallel(self.model)
    self.model.cuda()


    start_lr = self.cfg['start_lr']/64*self.cfg['batch_size']
    if self.cfg['optimizer_type']=='adam':
      self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, weight_decay=self.cfg['weight_decay'], betas=(0.9, 0.99), amsgrad=False)
    elif self.cfg['optimizer_type']=='sgd':
      self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr,weight_decay=self.cfg['weight_decay'], momentum=0.9)

    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg['lr_milestones'], gamma=0.1)


  def train_loop(self):
    self.model.train()
    model_fn = model_fn_decorator(test=False)

    avg_loss = {}
    for iter, batch in enumerate(self.train_loader):
      loss, preds, visual_dict, meter_dict = model_fn(batch, self.model, self.epoch)

      for k in visual_dict.keys():
        if k not in avg_loss:
          avg_loss[k] = []
        avg_loss[k].append(visual_dict[k])

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if iter%max(1,len(self.train_loader)//10)==0:
        print('epoch={}, {}/{}, train_loss={}'.format(self.epoch, iter, len(self.train_loader), loss.item()))

    for k in avg_loss.keys():
      avg_loss[k] = np.array(avg_loss[k]).mean()

    if avg_loss['loss']<self.best_train:
      self.best_train = avg_loss['loss']
      checkpoint_data = {'epoch': self.epoch, 'state_dict': self.model.state_dict(), 'best_res': self.best_train}
      dir = "{}/best_train.pth.tar".format(self.cfg['save_dir'])
      torch.save(checkpoint_data, dir,_use_new_zipfile_serialization=False)


  def val_loop(self):
    self.model.eval()
    model_fn = model_fn_decorator(test=False)

    avg_loss = {}
    with torch.no_grad():
      for iter,batch in enumerate(self.val_loader):
        loss, preds, visual_dict, meter_dict = model_fn(batch, self.model, self.epoch)

        for k in visual_dict.keys():
          if k not in avg_loss:
            avg_loss[k] = []
          avg_loss[k].append(visual_dict[k])

        if iter%max(1,len(self.val_loader)//10)==0:
          print('epoch={}, {}/{}, val_loss={}'.format(self.epoch,iter,len(self.val_loader),loss.item()))

    for k in avg_loss.keys():
      avg_loss[k] = np.array(avg_loss[k]).mean()

    if avg_loss['loss']<self.best_val:
      self.best_val = avg_loss['loss']
      checkpoint_data = {'epoch': self.epoch, 'state_dict': self.model.state_dict(), 'best_res': self.best_val}
      dir = "{}/best_val.pth.tar".format(self.cfg['save_dir'])
      torch.save(checkpoint_data, dir,_use_new_zipfile_serialization=False)


  def train(self):
    for self.epoch in range(self.cfg['n_epochs']):
      np.random.seed(self.cfg['random_seed']+self.epoch)
      print('epoch {}/{}'.format(self.epoch, self.cfg['n_epochs']))

      begin = time.time()
      self.train_loop()
      print("train loop time: {} s".format(time.time()-begin))
      print(">>>>>>>>>>>>>>>>>>>>")

      begin = time.time()
      self.val_loop()
      print("val loop time: {} s".format(time.time()-begin))
      print(">>>>>>>>>>>>>>>>>>>>")

      self.scheduler.step()
      torch.cuda.empty_cache()



