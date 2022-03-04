import numpy as np
from transformations import *
import os,sys,yaml,copy,pickle
import pybullet as p
import pybullet_data
from PIL import Image
from utils_pybullet import *
import utils_pybullet

gl_in_cv = np.eye(4)
gl_in_cv[1,1] = -1
gl_in_cv[2,2] = -1



class Camera:
  def __init__(self,K,H,W):
    self.H = H
    self.W = W
    self.K = K
    x0 = 0
    y0 = 0
    self.zfar = 100
    self.znear = 0.1
    self.projectionMatrix = \
      np.array([[2*K[0,0]/self.W,   -2*K[0,1]/self.W,       (self.W - 2*K[0,2] + 2*x0)/self.W,                  0],
              [          0,          2*K[1,1]/self.H,       (-self.H + 2*K[1,2] + 2*y0)/self.H,                0],
              [          0,             0, (-self.zfar - self.znear)/(self.zfar - self.znear), -2*self.zfar*self.znear/(self.zfar - self.znear)],
              [          0,             0,                             -1,                            0]]).reshape(4,4)
    cam_in_world = np.eye(4)
    cam_in_world[2,3] += 1e5
    code_dir = os.path.dirname(os.path.realpath(__file__))
    self.cam_id = create_object(f'{code_dir}/../data/object_models/kinectsensor.obj',scale=np.ones((3)),ob_in_world=cam_in_world,mass=0,has_collision=False)[0]

  def render(self,cam_in_world):
    ############ Show in GUI the camera pose
    set_body_pose_in_world(self.cam_id,cam_in_world)

    world_in_cam = np.linalg.inv(cam_in_world)
    world_in_cam_gl = gl_in_cv@world_in_cam
    _,_,rgb,depth,seg = p.getCameraImage(self.W,self.H,viewMatrix=world_in_cam_gl.T.reshape(-1),projectionMatrix=self.projectionMatrix.T.reshape(-1),shadow=1,lightDirection=[1, 1, 1])
    depth = self.zfar * self.znear / (self.zfar - (self.zfar - self.znear) * depth)
    depth[seg<0] = 0
    rgb = rgb[...,:3]
    return rgb,depth,seg