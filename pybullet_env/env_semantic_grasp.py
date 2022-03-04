import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
import operator
import numpy as np
import sys,os,glob,re,time,copy,trimesh,logging,gzip,gc
logging.getLogger().setLevel(logging.FATAL)
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append('{}/../'.format(code_dir))
sys.path.append("{}/../ss-pybullet".format(code_dir))
import pickle
from dexnet.grasping.gripper import save_grasp_pose_mesh,RobotGripper
from dexnet.grasping.grasp_sampler import PointConeGraspSampler
import dexnet
from autolab_core import YamlConfig
import os
import pybullet_tools.kuka_primitives as kuka_primitives
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


class EnvSemanticGraspNoArm(EnvBase):
  def __init__(self,env_grasp,ob_dir,ob_pts=None,place_dir=None,gui=False):
    super().__init__(gui)
    self.env_grasp = env_grasp
    self.contact_pts = None
    self.contact_dists = None

    self.ob_pts = copy.deepcopy(ob_pts)
    urdf_dir = '/tmp/{}_{}.urdf'.format(uuid4(),os.path.basename(ob_dir))
    create_urdf_for_mesh(ob_dir,out_dir=urdf_dir,concave=True)
    self.ob_id = p.loadURDF(urdf_dir, [0, 0, 0], useFixedBase=False)
    p.changeDynamics(self.ob_id,-1,linearDamping=0.9,angularDamping=0.9,lateralFriction=0.9,spinningFriction=0.9,mass=0.1,collisionMargin=0.0001)

    urdf_dir = '/tmp/{}_{}.urdf'.format(uuid4(),os.path.basename(place_dir))
    create_urdf_for_mesh(place_dir,out_dir=urdf_dir,concave=True)
    self.place_id = p.loadURDF(urdf_dir, [0, 0, 0], useFixedBase=True)
    p.changeVisualShape(self.place_id,-1,rgbaColor=[1,1,1,1])
    p.changeDynamics(self.place_id,-1,lateralFriction=0.1,spinningFriction=0.1,collisionMargin=0.0001)

    self.class_name = get_class_name(ob_dir)
    print('self.class_name',self.class_name)

    self.place_pose = np.eye(4)
    self.place_pose[:3,3] = np.array([0.7,0.,0.])
    if self.class_name=='hnm':
      rot = euler_matrix(np.pi,0,0,axes='sxyz')
      self.place_pose = rot@self.place_pose
    elif self.class_name=='screw':
      rot = euler_matrix(np.pi,0,0,axes='sxyz')
      self.place_pose = rot@self.place_pose
    set_body_pose_in_world(self.place_id,self.place_pose)
    p.setGravity(0,0,0)
    self.env_grasp.open_gripper()

    self.init_ob_in_world = self.place_pose@place_pose_dict[self.class_name][0]
    set_body_pose_in_world(self.ob_id,self.init_ob_in_world)

    self.state_id = p.saveState()


  def try_grasp(self,grasp_in_ob,debug=0):
    p.setGravity(0,0,0)
    PU.remove_fixed_constraint(self.ob_id,self.env_grasp.gripper_id,-1)
    p.restoreState(stateId=self.state_id)
    tmp = np.eye(4)
    tmp[:3,3] = [999,0,0]
    set_body_pose_in_world(self.place_id,tmp)

    set_body_pose_in_world(self.ob_id,self.init_ob_in_world)

    self.env_grasp.open_gripper()

    ob_in_world = get_ob_pose_in_world(self.ob_id)
    grasp_in_world = ob_in_world@grasp_in_ob
    self.env_grasp.set_gripper_pose_from_grasp_pose(grasp_in_world)

    if debug>=1:
      print('before close')
      time.sleep(1)
    sleep = 0
    if debug>=1:
      sleep = 0.1
    self.env_grasp.close_gripper(sleep=sleep,step=30)
    if debug>=1:
      print('after close')
      time.sleep(1)

    tmp_id = p.saveState()
    p.setGravity(0,0,-10)
    for _ in range(100):
      p.stepSimulation()
      if debug>=1:
        time.sleep(0.1)

    ob_in_world = get_ob_pose_in_world(self.ob_id)
    if ob_in_world[2,3]<-0.1 or np.linalg.norm(ob_in_world[:3,3]-grasp_in_world[:3,3])>0.2:
      p.removeState(tmp_id)
      if debug>=1:
        print('grasp failed')
        time.sleep(1)
      return 0

    p.setGravity(0,0,0)
    p.restoreState(tmp_id)
    p.removeState(tmp_id)

    self.contact_pts,self.contact_dists,n_side = self.env_grasp.get_grasp_contact_area(self.ob_id,self.ob_pts,get_pt_on_ob=True,surface_tol=0.002)
    if self.contact_pts is None or n_side<2:
      return 0

    set_body_pose_in_world(self.place_id,self.place_pose)
    attachment = PU.create_attachment(self.env_grasp.gripper_id,0,self.ob_id)
    ob_in_gripper = get_pose_A_in_B(self.ob_id,-1,self.env_grasp.gripper_id,-1)
    gripper_in_world = self.init_ob_in_world@np.linalg.inv(ob_in_gripper)
    set_body_pose_in_world(self.env_grasp.gripper_id,gripper_in_world)
    attachment.assign()
    if debug>=1:
      print('before go down')
      time.sleep(1)
    place_in_world = get_ob_pose_in_world(self.place_id)
    target_ob_in_world = place_in_world@place_pose_dict[self.class_name][1]
    target_gripper_in_world = target_ob_in_world@np.linalg.inv(ob_in_gripper)

    for gripper_pose in PU.interpolate_poses_matrix(gripper_in_world,target_gripper_in_world):
      set_body_pose_in_world(self.env_grasp.gripper_id,gripper_pose)
      attachment.assign()
      if debug>=1:
        time.sleep(0.1)
      if PU.any_link_pair_collision(body1=self.env_grasp.gripper_id,links1=self.env_grasp.finger_ids,body2=self.place_id):
        if debug>=1:
          print('collision')
          time.sleep(1)
        return 1

    if debug>=1:
      print('before remove hand')
      time.sleep(1)

    tmp = np.eye(4)
    tmp[0,3] = 100
    set_body_pose_in_world(self.env_grasp.gripper_id,tmp)

    if debug>=1:
      print('place action done, remove gripper, let it drop')
      time.sleep(1)

    p.setGravity(0,0,-10)
    for _ in range(50):
      p.stepSimulation()
      if debug>=1:
        time.sleep(0.1)

    place_success_func = get_place_success_func(self.class_name)
    ob_pose = get_ob_pose_in_world(self.ob_id)

    success = place_success_func(ob_pose,self.place_pose)
    if debug>=1:
      print(f'drop done, success={success}')
      time.sleep(1)

    if not success:
      return 1

    return 2



def generate_affordance_worker(grasps,symmetry_tfs,ob_pts,ob_dir,place_dir,gripper,gui,id,d,debug=False):
  results = []
  env_grasp = EnvGrasp(gripper,gui=gui)
  env = EnvSemanticGraspNoArm(env_grasp,ob_dir,ob_pts=ob_pts,place_dir=place_dir,gui=gui)

  class_name = get_class_name(ob_dir)
  pts_task_success = np.zeros((len(ob_pts))).astype(float)
  pts_grasp_success = np.zeros((len(ob_pts))).astype(float)
  kdtree = cKDTree(ob_pts)

  for i_grasp,grasp in enumerate(grasps):
    if i_grasp%max(1,(len(grasps)//10))==0:
      print('verify task oriented grasps {}/{}'.format(i_grasp,len(grasps)))

    grasp_in_ob = grasp.get_grasp_pose_matrix()

    ##########!NOTE For male objects, grasp from bottom definitely not work, dont waste time on it
    if class_name in ['hnm','screw']:
      found = False
      for symmetry_transform in symmetry_tfs:
        if np.linalg.det(symmetry_transform[:3,:3])<0:
          continue
        tmp_grasp_in_ob = symmetry_transform@grasp_in_ob
        approach_dir = tmp_grasp_in_ob[:3,0]
        approach_dir /= np.linalg.norm(approach_dir)
        dot = np.dot(approach_dir,np.array([0,0,1]))
        if dot<0:
          continue
        grasp_in_ob = copy.deepcopy(tmp_grasp_in_ob)
        found = True
        break
      if not found:
        continue

    ret = env.try_grasp(grasp_in_ob,debug=debug)
    results.append((grasp_in_ob,ret,env.contact_pts,env.contact_dists))

  del env,env_grasp
  d[id] = results


def generate_affordance(ob_pts,ob_normals,ob_dir,place_dir):
  '''Generate semantic grasp for one instance
  '''
  with gzip.open(f"{ob_dir.replace('.obj','_complete_grasp.pkl')}",'rb') as ff:
    grasps = pickle.load(ff)

  grasps = np.array(grasps)
  print(f"#grasps={len(grasps)}")
  grasps = np.random.choice(grasps,size=min(100000,len(grasps)),replace=False)

  symmetry_tfs = get_symmetry_tfs(class_name)

  N_CPU = multiprocessing.cpu_count()
  if debug>0:
    N_CPU = 1
  grasps_splits = np.array_split(grasps,N_CPU)
  manager = mp.Manager()
  d = manager.dict()
  workers = []
  for i in range(N_CPU):
    p = mp.Process(target=generate_affordance_worker, args=(grasps_splits[i],symmetry_tfs,ob_pts,ob_dir,place_dir,gripper,gui,i,d,debug))
    workers.append(p)
    p.start()

  results = []
  for i in range(N_CPU):
    workers[i].join()
    results += list(d[i])

  with gzip.open(ob_dir.replace('.obj','_affordance_results.pkl'),'wb') as ff:
    pickle.dump(results,ff)
  print(f"Write to {ob_dir.replace('.obj','_affordance_results.pkl')}")


def process_affordance_results(ob_pts,ob_normals,ob_dir):
  with gzip.open(ob_dir.replace('.obj','_affordance_results.pkl'),'rb') as ff:
    results = pickle.load(ff)

  ob_pts_score = np.zeros((len(ob_pts))).astype(float)
  pts_task_success = np.zeros((len(ob_pts))).astype(float)
  pts_grasp_success = np.zeros((len(ob_pts))).astype(float)

  ########### Merge results
  kdtree = cKDTree(ob_pts)
  grasp_center_map = {}
  for (grasp_in_ob,ret,contact_pts,contact_dists) in results:
    if ret==0:
      continue
    if contact_pts is None:
      continue
    dists,indices = kdtree.query(contact_pts)
    pts_grasp_success[indices] += 1
    if ret==2:
      pts_task_success[indices] += 1

    for id in indices:
      if id not in grasp_center_map:
        grasp_center_map[id] = []
      grasp_center_map[id].append((grasp_in_ob[:3,3],ret))

  print("pts_task_success: ",pts_task_success.round(3))
  print("pts_grasp_success: ",pts_grasp_success.round(3))

  min_valid_trial = 10
  colors = np.zeros(ob_pts.shape)
  ob_pts_score = pts_task_success/(pts_grasp_success+1e-10)
  ob_pts_score[pts_grasp_success<min_valid_trial] = 0.5
  score_max = ob_pts_score.max()
  score_min = ob_pts_score.min()
  print('score_min: {}, score_max: {}'.format(score_min,score_max))

  colors = array_to_heatmap_rgb(ob_pts_score)
  colors[pts_grasp_success<min_valid_trial] = [255,255,255]
  pcd = toOpen3dCloud(ob_pts,colors,ob_normals)
  o3d.io.write_point_cloud(ob_dir.replace('.obj','_affordance_vis.ply'),pcd)

  colors = np.zeros(ob_pts.shape)
  colors[:,0] = ob_pts_score
  pcd = toOpen3dCloud(ob_pts,colors,ob_normals)
  o3d.io.write_point_cloud(ob_dir.replace('.obj','_affordance.ply'),pcd)




if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--class_name',type=str,default='nut')
  parser.add_argument('--debug',type=int,default=0)
  args = parser.parse_args()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open('{}/../config.yml'.format(code_dir),'r') as ff:
    cfg = yaml.safe_load(ff)

  class_name = args.class_name
  debug = args.debug

  cfg_grasp = YamlConfig("{}/../config_grasp.yml".format(code_dir))
  gripper = RobotGripper.load(gripper_dir=cfg_grasp['gripper_dir'][class_name])

  if debug>0:
    gui = True
  else:
    gui = False

  ob_dirs = []
  names = cfg['dataset'][class_name]['train']
  code_dir = os.path.dirname(os.path.realpath(__file__))
  for name in names:
    ob_dirs.append(f'{code_dir}/../data/object_models/{name}')

  for ob_dir in ob_dirs:
    place_dir = ob_dir.replace('.obj','_place.obj')

    mesh = trimesh.load(ob_dir)
    ob_pts, face_ids = trimesh.sample.sample_surface_even(mesh,count=20000,radius=0.0005)
    ob_normals = mesh.face_normals[face_ids]
    pcd = toOpen3dCloud(ob_pts,normals=ob_normals)
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    ob_pts = np.asarray(pcd.points).copy()
    ob_normals = np.asarray(pcd.normals).copy()

    generate_affordance(ob_pts,ob_normals,ob_dir,place_dir)
    process_affordance_results(ob_pts,ob_normals,ob_dir)

