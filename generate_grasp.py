import warnings
warnings.filterwarnings("ignore")
import operator
import numpy as np
import sys,os,glob,re,time,copy,trimesh,gzip
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("{}/../".format(code_dir))
import pickle
from dexnet.grasping.gripper import RobotGripper
from dexnet.grasping.grasp_sampler import PointConeGraspSampler
import dexnet
from autolab_core import YamlConfig
import os,my_cpp
import multiprocessing
import matplotlib.pyplot as plt
from Utils import *
from multiprocessing import Pool
import multiprocessing
from multiprocessing import Process, Manager
from functools import partial
from itertools import repeat
try:
  multiprocessing.set_start_method('spawn')
except:
  pass
from pybullet_env.env import Env
from pybullet_env.env_grasp import *
from pybullet_env.utils_pybullet import *



def compute_grasp_score_worker(grasps,ob_dir,gripper,id,d,debug):
  if debug:
    gui = True
  else:
    gui = False
  env = EnvGrasp(gripper,gui=gui)
  env.add_obj(ob_dir,concave=True)
  p.changeDynamics(env.ob_id,-1,lateralFriction=0.7,spinningFriction=0.7,mass=0.1,collisionMargin=0.0001)
  p.setGravity(0,0,-10)
  for i_grasp in range(len(grasps)):
    if i_grasp%max(1,(len(grasps)//100))==0:
      print("compute_grasp_score_worker {}/{}".format(i_grasp,len(grasps)))
    grasp_pose = grasps[i_grasp].get_grasp_pose_matrix()
    grasps[i_grasp].perturbation_score = env.compute_perturbation_score(grasp_pose,trials=50)
  del env
  d[id] = grasps


def generate_grasp_one_object_balanced_score_from_complete_grasp(obj_dir):
  with gzip.open(obj_dir.replace('.obj','_complete_grasp.pkl'),'rb') as ff:
    grasps = pickle.load(ff)

  grasp_amount_per_bin = 1000
  grasp_score_bins = cfg_grasp['classes']
  n_grasp_score_bins = len(grasp_score_bins)-1
  n_grasps_per_bin = np.zeros((n_grasp_score_bins),dtype=int)
  grasp_bins = {}
  for i_bin in range(0,n_grasp_score_bins):
    grasp_bins[i_bin] = []

  for grasp in grasps:
    score = grasp.perturbation_score
    score_bin = np.digitize(score,grasp_score_bins) - 1
    grasp_bins[score_bin].append(grasp)

  for i_bin in grasp_bins.keys():
    grasp_bins[i_bin] = np.array(grasp_bins[i_bin])
    grasp_bins[i_bin] = np.random.choice(grasp_bins[i_bin],size=min(grasp_amount_per_bin,len(grasp_bins[i_bin])),replace=False)

  good_grasp = []
  for i_bin,grasps in grasp_bins.items():
    good_grasp += list(grasps)
  print("#grasp={}".format(len(good_grasp)))

  out_file = obj_dir.replace('.obj','_grasp_balanced_score.pkl')
  with gzip.open(out_file, 'wb') as f:
    pickle.dump(good_grasp, f)


def generate_grasp_one_object_complete_space(obj_dir):
  ags = PointConeGraspSampler(gripper,cfg_grasp)
  out_file = obj_dir.replace('.obj','_complete_grasp.pkl')

  mesh = trimesh.load(obj_dir)
  pts,face_ids = trimesh.sample.sample_surface_even(mesh,count=10000,radius=0.001)
  normals = mesh.face_normals[face_ids]
  pcd = toOpen3dCloud(pts,normals=normals)
  max_xyz = pts.max(axis=0)
  min_xyz = pts.min(axis=0)
  diameter = np.linalg.norm(max_xyz-min_xyz)

  pcd = pcd.voxel_down_sample(voxel_size=diameter/10.0)
  points_for_sample = np.asarray(pcd.points).copy()
  normals_for_sample = np.asarray(pcd.normals).copy()

  grasps = ags.sample_grasps(background_pts=np.ones((1,3))*99999,points_for_sample=points_for_sample,normals_for_sample=normals_for_sample,num_grasps=np.inf,max_num_samples=np.inf,n_sphere_dir=30,approach_step=0.005,ee_in_grasp=np.eye(4),cam_in_world=np.eye(4),upper=np.ones((7))*999,lower=-np.ones((7))*999,open_gripper_collision_pts=np.ones((1,3))*999999,center_ob_between_gripper=True,filter_ik=False,filter_approach_dir_face_camera=False,adjust_collision_pose=False)

  # grasps = [ParallelJawPtGrasp3D(grasp_pose=np.eye(4))]*20

  print(f'Evaluating #grasps={len(grasps)}')

  if debug:
    N_CPU = 1
  else:
    N_CPU = multiprocessing.cpu_count()
  grasps_split = np.array_split(grasps,N_CPU)
  manager = mp.Manager()
  d = manager.dict()
  workers = []
  for i in range(N_CPU):
    p = mp.Process(target=compute_grasp_score_worker, args=(grasps_split[i],obj_dir,gripper,i,d,debug))
    workers.append(p)
    p.start()

  grasps = []
  for i in range(N_CPU):
    workers[i].join()
    grasps += list(d[i])

  print(f"Saving #grasps={len(grasps)} to {out_file}")
  with gzip.open(out_file, 'wb') as f:
    pickle.dump(grasps, f)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--class_name',type=str,default='nut')
  parser.add_argument('--debug',type=int,default=0)
  args = parser.parse_args()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)

  class_name = args.class_name
  debug = args.debug

  cfg_grasp = YamlConfig("{}/config_grasp.yml".format(code_dir))
  gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir'][class_name])

  names = cfg['dataset'][class_name]['train']
  obj_dirs = []
  code_dir = os.path.dirname(os.path.realpath(__file__))
  for name in names:
    obj_dirs.append(f'{code_dir}/data/object_models/{name}')
  print("obj_dirs:\n",'\n'.join(obj_dirs))

  for obj_dir in obj_dirs:
    print('obj_dir',obj_dir)
    generate_grasp_one_object_complete_space(obj_dir)
    generate_grasp_one_object_balanced_score_from_complete_grasp(obj_dir)
