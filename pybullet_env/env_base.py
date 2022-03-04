import numpy as np
from transformations import *
import os,sys,yaml,copy,pickle,time,cv2,socket,argparse
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append("{}/../ss-pybullet".format(code_dir))
sys.path.append("{}/../".format(code_dir))
from Utils import *
import pybullet as p
import pybullet_data
import pybullet_tools.utils as PU
from PIL import Image
from utils_pybullet import *
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
try:
  multiprocessing.set_start_method('spawn')
except:
  pass

hostname = socket.gethostname()


class EnvBase:
  def __init__(self,gui=False):
    if not p.isConnected():
      if gui:
        self.client_id = p.connect(p.GUI)
      else:
        self.client_id = p.connect(p.DIRECT)
    else:
      print('bullet server already connected')

    self.gui = gui
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    code_dir = os.path.dirname(os.path.realpath(__file__))
    p.setAdditionalSearchPath(f'{code_dir}/../urdf')
    self.id_to_obj_file = {}
    self.id_to_scales = {}


  def __del__(self):
    try:
      p.disconnect()
      print("pybullet disconnected")
    except Exception as e:
      pass

  def reset(self):
    body_ids = PU.get_bodies()
    for body_id in body_ids:
      if body_id in self.env_body_ids:
        continue
      p.removeBody(body_id)
    body_ids = list(self.id_to_obj_file.keys())
    for body_id in body_ids:
      if body_id in self.env_body_ids:
        continue
      del self.id_to_obj_file[body_id]
      del self.id_to_scales[body_id]
