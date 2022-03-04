import numpy as np
from transformations import *
import os,sys,yaml,copy,pickle,struct,trimesh
from uuid import uuid4
import pybullet as p
import pybullet_data
from PIL import Image
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../')
sys.path.append('{}/../ss-pybullet'.format(code_dir))
import pybullet_tools.utils as PU

def get_bodies():
  return [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]



def add_gravity_to_ob(body_id,link_id=-1,gravity=-10):
  ob_mass = p.getDynamicsInfo(body_id,link_id)[0]
  p.applyExternalForce(body_id,link_id,forceObj=[0,0,gravity*ob_mass],posObj=[0,0,0],flags=p.LINK_FRAME)


def print_all_joints(robot_id):
  n_joints = p.getNumJoints(robot_id)
  for joint_id in range(n_joints):
    print(p.getJointInfo(robot_id,joint_id))


def get_ob_pose_in_world(body_id):
  trans,q_xyzw = p.getBasePositionAndOrientation(body_id)
  ob_in_world = np.eye(4)
  ob_in_world[:3,3] = trans
  q_wxyz = [q_xyzw[-1],q_xyzw[0],q_xyzw[1],q_xyzw[2]]
  R = quaternion_matrix(q_wxyz)[:3,:3]
  ob_in_world[:3,:3] = R
  return ob_in_world


def get_link_pose_in_world(body_id,link_id):
  trans,rot = p.getLinkState(body_id,link_id)[4:6]
  link_in_world = np.eye(4)
  link_in_world[:3,3] = trans
  q_wxyz = [rot[3], *rot[:3]]
  link_in_world[:3,:3] = quaternion_matrix(q_wxyz)[:3,:3]
  return link_in_world

def get_pose_A_in_B(bodyA,linkA,bodyB,linkB):
  if linkA==-1:
    A_in_world = get_ob_pose_in_world(bodyA)
  else:
    A_in_world = get_link_pose_in_world(bodyA,linkA)
  if linkB==-1:
    B_in_world = get_ob_pose_in_world(bodyB)
  else:
    B_in_world = get_link_pose_in_world(bodyB,linkB)
  A_in_B = np.linalg.inv(B_in_world)@A_in_world
  return A_in_B



def set_body_pose_in_world(body_id,ob_in_world):
  trans = ob_in_world[:3,3]
  q_wxyz = quaternion_from_matrix(ob_in_world)
  q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
  p.resetBasePositionAndOrientation(body_id,trans,q_xyzw)



def create_urdf_for_mesh(mesh_dir,concave=False, out_dir=None, mass=0.1, has_collision=True, scale=np.ones((3))):
  assert '.obj' in mesh_dir, f'mesh_dir={mesh_dir}'

  lateral_friction = 0.8
  spinning_friction = 0.5
  rolling_friction = 0.5

  concave_str = 'no'
  collision_mesh_dir = copy.deepcopy(mesh_dir)
  if concave:
    concave_str = 'yes'
    if mass!=0:
      collision_mesh_dir = mesh_dir.replace('.obj','_vhacd.obj')

  collision_block = ""
  if has_collision:
    collision_block = f"""
      <collision>
        <origin xyz="0 0 0"/>
        <geometry>
          <mesh filename="{collision_mesh_dir}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
        </geometry>
      </collision>
      """

  link_str = f"""
    <link concave="{concave_str}" name="base_link">
      <inertial>
        <origin xyz="0 0 0" />
        <mass value="{mass}" />
        <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0"/>
        <geometry>
          <mesh filename="{mesh_dir}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
        </geometry>
      </visual>
      {collision_block}
    </link>
  """

  urdf_str = f"""
  <robot name="model.urdf">
    {link_str}
  </robot>
  """

  if out_dir is None:
    out_dir = mesh_dir.replace('.obj','.urdf')
  with open(out_dir,'w') as ff:
    ff.write(urdf_str)

  return out_dir





def create_object(obj_file,scale,ob_in_world,mass,has_collision=True,useFixedBase=False,concave=False,collision_margin=0.0001):
  '''
  @scale: np array (3)
  '''
  urdf_dir = f'/tmp/{os.path.basename(obj_file)}_{uuid4()}.urdf'
  create_urdf_for_mesh(obj_file,out_dir=urdf_dir,mass=mass,has_collision=has_collision,concave=concave,scale=scale)
  q_wxyz = quaternion_from_matrix(ob_in_world)
  q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
  ob_id = p.loadURDF(urdf_dir, basePosition=ob_in_world[:3,3], baseOrientation=q_xyzw, useFixedBase=useFixedBase, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES|p.URDF_USE_MATERIAL_COLORS_FROM_MTL|p.URDF_MAINTAIN_LINK_ORDER)
  set_body_pose_in_world(ob_id,ob_in_world)
  p.changeDynamics(ob_id,-1,collisionMargin=collision_margin)
  return ob_id, urdf_dir


def create_duplicate_object(n_ob,obj_file,scale,ob_in_worlds,mass,has_collision=True,concave=False,useFixedBase=False,collision_margin=0.0001):
  ob_ids = []
  urdf_dir = None
  for i in range(n_ob):
    ob_in_world = ob_in_worlds[i]
    q_wxyz = quaternion_from_matrix(ob_in_world)
    q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
    if i==0:
      ob_id,urdf_dir = create_object(obj_file,scale=scale,ob_in_world=ob_in_world,mass=mass,has_collision=has_collision,concave=concave,useFixedBase=useFixedBase)
    else:
      ob_id = p.loadURDF(urdf_dir, basePosition=ob_in_world[:3,3], baseOrientation=q_xyzw, useFixedBase=useFixedBase,flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES|p.URDF_USE_MATERIAL_COLORS_FROM_MTL|p.URDF_MAINTAIN_LINK_ORDER)
    p.changeDynamics(ob_id,-1,collisionMargin=collision_margin)
    set_body_pose_in_world(ob_id,ob_in_world)
    ob_ids.append(ob_id)
  return ob_ids



def create_gripper_visual_shape(gripper,has_collision=False,mass=0):
  obj_file = gripper.mesh_filename
  urdf_dir = f'/tmp/gripper{uuid4()}.urdf'
  create_urdf_for_mesh(obj_file,out_dir=urdf_dir,has_collision=has_collision,mass=mass)
  gripper_id = p.loadURDF(urdf_dir,[0, 0, 0], useFixedBase=False)
  p.changeVisualShape(gripper_id,-1,rgbaColor=[1,0,0,1])
  return gripper_id




if __name__=="__main__":
  from pybullet_env.env_base import EnvBase
  env = EnvBase(gui=True)
  out_dir = '/tmp/1.urdf'


