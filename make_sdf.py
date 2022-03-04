import numpy as np
import os,sys,copy,glob,cv2,trimesh,time,shutil,pickle,gzip,logging,argparse,difflib
from sklearn.cluster import DBSCAN
logging.getLogger().setLevel(logging.FATAL)
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append('{}/ss-pybullet'.format(code_dir))
import warnings
warnings.filterwarnings("ignore")
import open3d as o3d
from PIL import Image
from transformations import *
from pybullet_env.utils_pybullet import *
from pybullet_env.env import *
from pybullet_env.env_grasp import *
import pybullet as p
import pybullet_data
from Utils import *
from data_reader import *
import pybullet_tools.utils as PU
import matplotlib.pyplot as plt
from dexnet.grasping.gripper import RobotGripper
from autolab_core import YamlConfig
from dexnet.grasping.gripper import save_grasp_pose_mesh
from renderer import ModelRendererOffscreen
import mpl_toolkits.mplot3d.axes3d as p3



def generate_sdf_worker(obj_file,resolution=0.001,padding=5):
  mesh = trimesh.load(obj_file)
  dimensions = mesh.vertices.max(axis=0)-mesh.vertices.min(axis=0)
  dim = np.ceil(dimensions.max()/resolution) + padding*2
  cmd = f"SDFGen {obj_file} {int(dim)} {int(padding)}"
  print(cmd)
  os.system(cmd)


def generate_sdf():
  names = cfg['dataset'][args.class_name]['train']+cfg['dataset'][args.class_name]['test']
  obj_files = []
  code_dir = os.path.dirname(os.path.realpath(__file__))
  for name in names:
    obj_files.append(f'{code_dir}/data/object_models/{name}')

  print('obj_files:')
  print('\n'.join(obj_files))

  for obj_file in obj_files:
    generate_sdf_worker(obj_file)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--class_name',type=str,default='nut')
  args = parser.parse_args()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)

  generate_sdf()
