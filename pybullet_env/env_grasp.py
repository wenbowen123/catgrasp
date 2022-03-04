import numpy as np
import open3d as o3d
from transformations import *
import os,sys,yaml,copy,pickle,time,cv2,socket,argparse,inspect,trimesh
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append("{}/../".format(code_dir))
sys.path.append("{}/../ss-pybullet".format(code_dir))
from dexnet.grasping.grasp import ParallelJawPtGrasp3D
from dexnet.grasping.gripper import save_grasp_pose_mesh
from autolab_core import YamlConfig
from dexnet.grasping.gripper import RobotGripper
from Utils import *
import pybullet as p
import pybullet_data
import pybullet_tools.utils as PU
from PIL import Image
from camera import *
from utils_pybullet import *
from env_base import EnvBase
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
try:
  multiprocessing.set_start_method('spawn')
except:
  pass

hostname = socket.gethostname()



class EnvGrasp(EnvBase):
  def __init__(self,gripper,gui=False):
    EnvBase.__init__(self,gui)

    self.id_to_obj_file = {}
    self.id_to_scales = {}

    code_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{code_dir}/../config_grasp.yml','r') as ff:
      cfg_grasp = yaml.safe_load(ff)
    self.cfg_grasp = cfg_grasp

    self.gripper = gripper
    gripper_dir = gripper.gripper_folder
    print("gripper_dir",gripper_dir)
    finger_mesh1 = trimesh.load(f'{gripper_dir}/finger1.obj')
    finger_mesh2 = copy.deepcopy(finger_mesh1)
    R_z = euler_matrix(0,0,np.pi,axes='sxyz')
    finger_mesh2.apply_transform(R_z)
    self.finger_meshes = [finger_mesh1, finger_mesh2]

    gripper_urdf = f"{gripper_dir}/gripper.urdf"
    self.gripper_id = p.loadURDF(gripper_urdf, [0, 0, 0], useFixedBase=True)
    self.id_to_obj_file[self.gripper_id] = gripper_urdf
    self.finger_ids = np.array([1,2],dtype=int)
    self.gripper_max_force = np.ones((2),dtype=float)*100
    p.changeDynamics(self.gripper_id,-1,lateralFriction=0.9,spinningFriction=0.9)
    self.grasp_pose_in_gripper_base = self.gripper.get_grasp_pose_in_gripper_base()
    self.grip_dirs = np.array([[0,1,0],[0,-1,0]])


    self.env_body_ids = PU.get_bodies()
    print("self.env_body_ids",self.env_body_ids)


  def add_obj(self,ob_dir,concave=False):
    urdf_dir = '/tmp/{}_{}.urdf'.format(np.random.randint(99999),os.path.basename(ob_dir))
    create_urdf_for_mesh(ob_dir,out_dir=urdf_dir,concave=concave)
    self.ob_id = p.loadURDF(urdf_dir, [0, 0, 0], useFixedBase=False, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    self.id_to_obj_file[self.ob_id] = ob_dir


  def add_placeholder(self,place_ob_dir):
    urdf_dir = '/tmp/{}_{}.urdf'.format(np.random.randint(99999),os.path.basename(place_ob_dir))
    create_urdf_for_mesh(place_ob_dir,out_dir=urdf_dir,concave=True)
    self.place_id = p.loadURDF(urdf_dir, [0, 0, 0], useFixedBase=True)
    self.id_to_obj_file[self.place_id] = place_ob_dir
    place_in_world = np.eye(4)
    place_in_world[0,3] = 2
    set_body_pose_in_world(self.place_id,place_in_world)
    p.changeDynamics(self.place_id,-1,lateralFriction=0.1,spinningFriction=0.1,collisionMargin=0.0001)


  def verify_grasp(self,grasp_in_world):
    self.open_gripper()
    set_body_pose_in_world(self.ob_id,np.eye(4))
    self.set_gripper_pose_from_grasp_pose(grasp_in_world)
    if PU.body_collision(self.gripper_id,self.ob_id):
      return False

    self.close_gripper()

    ob_init_pose = get_ob_pose_in_world(self.ob_id)
    for _ in range(50):
      add_gravity_to_ob(self.ob_id,gravity=-10)
      p.stepSimulation()

    success = False
    ob_final_pose = get_ob_pose_in_world(self.ob_id)
    if np.linalg.norm(ob_init_pose[:3,3]-ob_final_pose[:3,3])<=0.02:
      success = True
    else:
      success = False

    return success


  def compute_perturbation_score(self,grasp_pose,trials=50):
    successes = []
    for _ in range(trials):
      offset_pose = random_uniform_magnitude(max_T=0.005,max_R=10)
      tmp_grasp_in_ob = grasp_pose@offset_pose
      success = self.verify_grasp(tmp_grasp_in_ob)
      successes.append(success)
    successes = np.array(successes)
    return np.sum(successes)/len(successes)


  def set_gripper_pose_from_grasp_pose(self,grasp_in_world):
    gripper_base_in_world = grasp_in_world@np.linalg.inv(self.grasp_pose_in_gripper_base)
    set_body_pose_in_world(self.gripper_id,gripper_base_in_world)


  def close_gripper(self,step=50,sleep=0):
    p.setJointMotorControlArray(self.gripper_id,jointIndices=self.finger_ids,controlMode=p.POSITION_CONTROL,targetPositions=[1,1],forces=self.gripper_max_force)
    for _ in range(step):
      p.stepSimulation()
      time.sleep(sleep)



  def get_grasp_contact_area(self,ob_id,ob_pts,get_pt_on_ob=True,surface_tol=0.002):
    ob_in_world = get_ob_pose_in_world(ob_id)
    out_pts = []
    out_dists = []
    for i_finger,finger_id in enumerate(self.finger_ids):
      finger_in_world = PU.get_link_mesh_pose_matrix(self.gripper_id,finger_id)
      finger_in_ob = np.linalg.inv(ob_in_world)@finger_in_world
      ob_in_finger = np.linalg.inv(finger_in_ob)
      surface_pts,dist_to_finger_surface = get_finger_contact_area(self.finger_meshes[i_finger],ob_in_finger,ob_pts,grip_dir=self.grip_dirs[i_finger],surface_tol=surface_tol)
      if surface_pts is None:
        continue

      out_pts.append(surface_pts)
      out_dists.append(dist_to_finger_surface)

    n_side = len(out_pts)
    if n_side==0:
      return None,None,0
    if n_side==2:
      out_pts = np.concatenate(out_pts,axis=0).reshape(-1,3)
      out_dists = np.concatenate(out_dists,axis=0).reshape(-1)
    else:
      out_pts = out_pts[0]
      out_dists = out_dists[0]

    return out_pts, out_dists, n_side


  def open_gripper(self,step=50):
    p.setJointMotorControlArray(self.gripper_id,jointIndices=self.finger_ids,controlMode=p.POSITION_CONTROL,targetPositions=[0,0],forces=self.gripper_max_force)
    for _ in range(step):
      p.stepSimulation()


def show_selected_grasps_with_color(ags, grasps, mesh_dir, title, gripper_mesh_dir=None, grasp_in_gripper_base=None, graspable=None, save_fig=False, show_fig=True, draw_contact=False):
  from mayavi import mlab
  max_n_grasp_plot = 3
  m_good = []
  m_bad = []
  grasps = np.array(grasps).reshape(-1)
  for grasp in grasps:
    if grasp.perturbation_score>=0.5:
      m_good.append(grasp)
    else:
      m_bad.append(grasp)
  print("#m_good={}".format(len(m_good)))
  print("#m_bad={}".format(len(m_bad)))
  m_good = np.array(m_good)
  m_bad = np.array(m_bad)
  if len(m_good)>0:
    m_good = m_good[np.random.choice(len(m_good), size=min(max_n_grasp_plot,len(m_good)), replace=False)]
  if len(m_bad)>0:
    m_bad = m_bad[np.random.choice(len(m_bad), size=min(max_n_grasp_plot,len(m_bad)), replace=False)]
  collision_grasp_num = 0
  mesh = trimesh.load(mesh_dir)
  if gripper_mesh_dir is not None:
    gripper_mesh = trimesh.load(gripper_mesh_dir)
  else:
    gripper_mesh = None

  if save_fig or show_fig:
    if len(m_good)>0:
      mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
      mlab.triangular_mesh(mesh.vertices[:,0],mesh.vertices[:,1],mesh.vertices[:,2],mesh.faces,color=(0.5,0.5,0.5),opacity=0.8)
      for a in m_good:
        collision_free = display_grasp(ags, a, graspable=graspable, color=(0,1,0), gripper_mesh=gripper_mesh, grasp_in_gripper_base=grasp_in_gripper_base, show_fig=show_fig, save_fig=save_fig, draw_contact=draw_contact)  # simulated gripper
        if not collision_free:
          collision_grasp_num += 1

      if save_fig:
        mlab.savefig("good_"+title+".png")
        mlab.close()
      elif show_fig:
        mlab.title(title, size=0.5)

    if len(m_bad)>0:
      mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
      mlab.triangular_mesh(mesh.vertices[:,0],mesh.vertices[:,1],mesh.vertices[:,2],mesh.faces,color=(0.5,0.5,0.5),opacity=0.8)

      for a in m_bad:
        collision_free = display_grasp(ags, a, graspable=graspable, color=(1, 0, 0), gripper_mesh=gripper_mesh, grasp_in_gripper_base=grasp_in_gripper_base, show_fig=show_fig, save_fig=save_fig, draw_contact=draw_contact)
        if not collision_free:
          collision_grasp_num += 1

      if save_fig:
        mlab.savefig("bad_"+title+".png")
        mlab.close()
      elif show_fig:
        mlab.title(title, size=0.5)

    if show_fig:
      mlab.show()

  elif generate_new_file:
    collision_grasp_num = 0
    ind_good_grasp_ = []
    for i_ in range(len(m)):
      collision_free = display_grasp(ags, m[i_][0], graspable=graspable, color=(1, 0, 0), show_fig=show_fig, save_fig=save_fig)
      if not collision_free:
        collision_grasp_num += 1
      else:
        ind_good_grasp_.append(i_)
    collision_grasp_num = str(collision_grasp_num)
    collision_grasp_num = (4-len(collision_grasp_num))*" " + collision_grasp_num
    print("collision_grasp_num =", collision_grasp_num, "| object name:", title)
    return ind_good_grasp_


def get_finger_contact_area(finger_mesh,ob_in_finger,ob_pts,grip_dir,ob_normals=None,surface_tol=0.002,debug=False):
    grip_dir = np.array(grip_dir)
    grip_dir = grip_dir/np.linalg.norm(grip_dir)
    pcd = toOpen3dCloud(ob_pts,normals=ob_normals)
    pcd.transform(ob_in_finger)

    cur_ob_pts = np.asarray(pcd.points).copy()
    if ob_normals is not None:
      cur_ob_normals = np.asarray(pcd.normals).copy()
    within_finger_mask = (cur_ob_pts[:,0]>=finger_mesh.vertices[:,0].min()) & (cur_ob_pts[:,0]<=finger_mesh.vertices[:,0].max()) & (cur_ob_pts[:,2]>=finger_mesh.vertices[:,2].min()) & (cur_ob_pts[:,2]<=finger_mesh.vertices[:,2].max())
    if within_finger_mask.sum()==0:
      return None,None

    within_finger_pts = cur_ob_pts[within_finger_mask]
    if ob_normals is not None:
      within_finger_normals = cur_ob_normals[within_finger_mask]

    if np.allclose(grip_dir,np.array([0,1,0])):
      dist_to_finger_surface = np.abs(within_finger_pts[:,1]-within_finger_pts[:,1].min())  #The bottom points touches finger firstly
    elif np.allclose(grip_dir,np.array([0,-1,0])):
      dist_to_finger_surface = np.abs(within_finger_pts[:,1]-within_finger_pts[:,1].max())
    else:
      raise RuntimeError(f'grip_dir={grip_dir}')

    contact_mask = dist_to_finger_surface<=surface_tol
    if contact_mask.sum()==0:
      return None,None

    dist_to_finger_surface = dist_to_finger_surface[contact_mask]
    surface_pts = within_finger_pts[contact_mask]
    if ob_normals is not None:
      surface_normals = within_finger_normals[contact_mask]
      closest_id = np.abs(dist_to_finger_surface).argmin()
      closest_normal = surface_normals[closest_id]
      closest_normal /= np.linalg.norm(closest_normal)
      dot = np.dot(closest_normal,grip_dir)
      if dot>0:
        return None,None

    surface_pts = (np.linalg.inv(ob_in_finger)@to_homo(surface_pts).T).T[:,:3]
    return surface_pts, dist_to_finger_surface
