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
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
try:
  multiprocessing.set_start_method('spawn')
except:
  pass



def compute_canonical_model():
  obj_files = []
  names = cfg['dataset'][class_name]['train']
  code_dir = os.path.dirname(os.path.realpath(__file__))

  for name in names:
    obj_files.append(f'{code_dir}/data/object_models/{name}')
  out_file = f'{code_dir}/data/object_models/{class_name}_canonical.pkl'

  print("obj_files:\n", '\n'.join(obj_files))

  clouds = {}
  file2normals = {}
  for file in obj_files:
    mesh = trimesh.load(file)
    pts, face_ids = trimesh.sample.sample_surface_even(mesh, 20000)
    normals = mesh.face_normals[face_ids]
    file2normals[file] = copy.deepcopy(normals)
    pcd = toOpen3dCloud(pts)
    pcd.voxel_down_sample(voxel_size=0.001)
    clouds[file] = np.asarray(pcd.points).copy()

  transforms_to_nocs = {}
  for obj_file in obj_files:
    max_xyz = clouds[obj_file].max(axis=0)
    min_xyz = clouds[obj_file].min(axis=0)
    transforms_to_nocs[obj_file] = np.eye(4)
    nunocs_scale = 1.
    center = (max_xyz+min_xyz)/2
    transforms_to_nocs[obj_file][:3,3] = -center
    transforms_to_nocs[obj_file][:3,:3] = np.diag(np.ones((3))/(max_xyz-min_xyz)) / nunocs_scale
    pcd = toOpen3dCloud(clouds[obj_file])
    pcd.transform(transforms_to_nocs[obj_file])
    max_xyz = np.asarray(pcd.points).max(axis=0)
    min_xyz = np.asarray(pcd.points).min(axis=0)
    new_tf = np.eye(4)
    new_tf[:3,3] = -(max_xyz+min_xyz)/2
    transforms_to_nocs[obj_file] = new_tf@transforms_to_nocs[obj_file]

  dist_to_other_models = {}
  for i,obj_file in enumerate(obj_files):
    dists = []
    cloud = copy.deepcopy(clouds[obj_file])
    cloud = (transforms_to_nocs[obj_file]@to_homo(cloud).T).T[:,:3]
    for other_file in obj_files:
      if obj_file==other_file:
        continue
      other_cloud = copy.deepcopy(clouds[other_file])
      other_cloud = (transforms_to_nocs[other_file]@to_homo(other_cloud).T)[:,:3]
      cd = chamfer_distance_between_clouds_mutual(cloud,other_cloud)
      dists.append(cd)
    avg_dist = np.concatenate(dists,axis=0).reshape(-1).mean()
    dist_to_other_models[obj_file] = avg_dist

  best_dist_id = np.array(dist_to_other_models.values()).argmin()
  best_file = list(dist_to_other_models.keys())[best_dist_id]
  print('best_file',best_file)
  canonical_cloud = copy.deepcopy(clouds[best_file])
  canonical_normals = copy.deepcopy(file2normals[best_file])
  pcd = toOpen3dCloud(canonical_cloud,normals=canonical_normals)
  pcd.transform(transforms_to_nocs[best_file])
  canonical_cloud = np.asarray(pcd.points).copy()
  canonical_normals = np.asarray(pcd.normals).copy()

  ############ Gather grasp codebook
  gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir'][class_name])
  canonical_grasps = []
  grasp_score_thres = 0.8
  for obj_file in obj_files:
    grasp_file = obj_file.replace('.obj','_complete_grasp.pkl')
    with gzip.open(grasp_file,'rb') as ff:
      grasps = pickle.load(ff)
    new_grasps = []
    print(f"Before filter score, #grasps={len(grasps)}")
    for grasp in grasps:
      if grasp.perturbation_score<grasp_score_thres:
        continue
      new_grasps.append(grasp)
    grasps = new_grasps
    grasps = np.array(grasps)
    print(f"After filter score, #grasps={len(grasps)}")
    for grasp in grasps:
      canonical_grasp = copy.deepcopy(grasp)
      grasp_pose = grasp.get_grasp_pose_matrix()
      canonical_grasp.grasp_pose = transforms_to_nocs[obj_file]@grasp_pose
      canonical_grasp.grasp_pose = normalizeRotation(canonical_grasp.grasp_pose)
      canonical_grasps.append(canonical_grasp)
  canonical_grasps = np.array(canonical_grasps)

  ############ Gather affordance codebook
  print("Gathering affordance")
  canonical_affordance = np.zeros((len(canonical_cloud)))
  n_afford = 0
  for obj_file in obj_files:
    print("obj_file:",obj_file)
    affordance_file = obj_file.replace('.obj','_affordance.ply')
    affordance_pcd = o3d.io.read_point_cloud(affordance_file)
    affordance_pcd.transform(transforms_to_nocs[obj_file])
    afforance_pts = np.asarray(affordance_pcd.points).copy()
    affordance_probs = np.asarray(affordance_pcd.colors)[:,0].copy()
    kdtree = cKDTree(afforance_pts)
    print("affordance_probs",affordance_probs.min(),affordance_probs.max())
    dists,indices = kdtree.query(canonical_cloud)
    canonical_affordance += affordance_probs[indices]
    n_afford += 1
  canonical_affordance /= n_afford

  afford_max = canonical_affordance.max()
  afford_min = canonical_affordance.min()
  print("canonical afford_min: {}, afford_max: {}".format(afford_min,afford_max))
  colors = array_to_heatmap_rgb(canonical_affordance.reshape(-1))
  pcd = toOpen3dCloud(canonical_cloud,colors)
  o3d.io.write_point_cloud(out_file.replace('.pkl','_affordance_vis.ply'),pcd,write_ascii=True)

  print(f"Write to {out_file}")
  with gzip.open(out_file,'wb') as ff:
    out = {
      'obj_files': obj_files,
      'canonical_cloud': canonical_cloud,
      'canonical_normals': canonical_normals,
      'transforms_to_nocs': transforms_to_nocs,
      'canonical_grasps': canonical_grasps,
      'canonical_affordance': canonical_affordance,
    }
    print(f'#canonical_grasps={len(canonical_grasps)}')
    pickle.dump(out,ff)




if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--class_name',type=str,default='nut')
  args = parser.parse_args()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open('{}/config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)
  cfg_grasp = YamlConfig("{}/config_grasp.yml".format(code_dir))

  class_name = args.class_name
  gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir'][class_name])

  compute_canonical_model()