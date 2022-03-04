import numpy as np
from transformations import *
import xml.etree.ElementTree as ET
import os,sys,yaml,copy,pickle,time,cv2,socket,argparse,gzip,trimesh
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append("{}/../ss-pybullet".format(code_dir))
import pybullet_tools.kuka_primitives as kuka_primitives
sys.path.append("{}/../".format(code_dir))
from Utils import *
import pybullet as p
import pybullet_data
import pybullet_tools.utils as PU
from PIL import Image
from camera import *
from utils_pybullet import *
from dexnet.grasping.gripper import RobotGripper
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
try:
  multiprocessing.set_start_method('spawn')
except:
  pass
from env_base import EnvBase
from env_grasp import EnvGrasp
hostname = socket.gethostname()



class Env(EnvBase):
  def __init__(self,cfg,gripper,gui=False):
    from ikfast_pybind import get_ik_iiwa14
    EnvBase.__init__(self,gui)
    self.cfg = cfg
    self.K = np.array(self.cfg['K']).reshape(3,3)
    H = self.cfg['H']
    W = self.cfg['W']
    self.camera = Camera(self.K,H,W)
    self.cam_in_world = np.array([-0.0841524825,   0.992909968, -0.0839533508,   0.607104242,
                                  0.981524885,  0.0680680275,  -0.178817391,    0.30635184,
                                -0.171835035, -0.0974502265,   -0.98029393,   0.705115497,
                                            0,             0,             0,             1]).reshape(4,4)
    set_body_pose_in_world(self.camera.cam_id,self.cam_in_world)
    self.bin_in_world = np.array([-8.72530932e-04,  1.00686988e+00, -1.40173813e-02,  5.58573728e-01,
                                -1.00689921e+00, -7.08176661e-04,  1.18124293e-02,  3.67436346e-01,
                                1.18008949e-02,  1.40269522e-02,  1.00680166e+00, -5.22509020e-02,
                                0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]).reshape(4,4)
    self.bin_in_cam = np.linalg.inv(self.cam_in_world)@self.bin_in_world
    self.cam_in_bin = np.linalg.inv(self.bin_in_cam)
    self.clutter_id = None
    self.env_grasp = EnvGrasp(gripper,gui=self.gui)
    ob_in_world = np.eye(4)
    ob_in_world[2,3] = 9999
    set_body_pose_in_world(self.env_grasp.gripper_id,ob_in_world)

    code_dir = os.path.dirname(os.path.realpath(__file__))
    robot_dir = f'{code_dir}/../urdf/robot.urdf'
    self.robot_id = p.loadURDF(robot_dir, [0, 0, 0], useFixedBase=True)
    set_body_pose_in_world(self.robot_id,np.eye(4))
    self.get_ik_func = get_ik_iiwa14
    self.id_to_obj_file[self.robot_id] = robot_dir
    self.id_to_scales[self.robot_id] = np.ones((3),dtype=float)
    self.arm_ids = PU.get_movable_joints(self.robot_id)[:7]
    self.env_grasp.gripper_id = self.robot_id
    self.gripper_id = 12
    self.ee_id = 7
    self.gripper_in_ee = get_pose_A_in_B(self.robot_id,self.gripper_id,self.robot_id,self.ee_id)
    self.grasp_in_ee = self.gripper_in_ee@self.env_grasp.grasp_pose_in_gripper_base
    self.env_grasp.finger_ids = np.array([13,14])
    self.finger_ids = self.env_grasp.finger_ids
    self.lower_limits, self.upper_limits = PU.get_joints_limits(self.robot_id,self.arm_ids)
    print_all_joints(self.robot_id)

    p.setGravity(0,0,-10)

    self.env_body_ids = PU.get_bodies()
    print("self.env_body_ids",self.env_body_ids)



  def add_table(self):
    table_top_center = np.array([0.6,0,0.5])
    code_dir = os.path.dirname(os.path.realpath(__file__))
    table_dir = f'{code_dir}/../data/object_models/table.urdf'
    self.table_id = p.loadURDF(table_dir, [table_top_center[0],table_top_center[1], 0],useFixedBase=True,flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    self.id_to_obj_file[self.table_id] = table_dir
    self.id_to_scales[self.robot_id] = np.ones((3),dtype=float)


  def add_bin(self,pos=None,scale=1):
    code_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = f'{code_dir}/../data/object_models/tray_textured_small_with_base.obj'
    ob_in_world = self.bin_in_world.copy()
    if pos is not None:
      ob_in_world[:3,3] = pos.reshape(3)
    self.bin_id,_ = create_object(bin_dir,scale=np.ones((3))*scale,ob_in_world=ob_in_world,mass=0.1,useFixedBase=True,concave=True)
    p.changeDynamics(self.bin_id,-1,collisionMargin=0.0001)
    p.changeVisualShape(self.bin_id,-1,rgbaColor=[0.5,0.5,0.5,1])
    self.id_to_obj_file[self.bin_id] = bin_dir
    self.id_to_scales[self.robot_id] = np.ones((3),dtype=float)*scale
    bin_verts = p.getMeshData(self.bin_id)[1]
    bin_verts = np.array(bin_verts).reshape(-1,3)
    self.bin_dimensions = bin_verts[:,:2].max(axis=0)-bin_verts[:,:2].min(axis=0)
    self.bin_in_world = get_ob_pose_in_world(self.bin_id)
    self.bin_verts = bin_verts


  def ik_fast_feasible_solutions(self,ee_link_pose,obstacles=[],attachments=[]):
    sols = self.get_ik_func(trans_list=ee_link_pose[:3,3], rot_list=ee_link_pose[:3,:3])
    feasible_sols = []

    tmp_id = p.saveState()

    for sol in sols:
      violate = PU.violates_limits(self.robot_id,self.arm_ids,sol)
      if violate:
        continue

      if len(obstacles)>0:
        PU.set_joint_positions(self.robot_id, self.arm_ids, sol)
        for attach in attachments:
          attach.assign()
        is_collision = PU.pairwise_collisions(self.robot_id,obstacles,debug=False)
        for attach in attachments:
          is_collision = is_collision or PU.pairwise_collisions(attach.child,obstacles)
        if is_collision:
          print(f'ik_fast_feasible_solutions collisoin')
          continue

      feasible_sols.append(sol)

    p.restoreState(tmp_id)
    p.removeState(tmp_id)

    return feasible_sols


  def move_arm(self,link_id,link_pose=None,joint_positions=None,ignore_joint_limits=False,obstacles=[],attachments=[],teleport=False,ignore_all_collsion=False,timeout=999999,use_ikfast=True,collision_fn=None,resolutions=None,check_end_collision=True):
    assert link_id in [self.gripper_id, self.ee_id], f"link_id={link_id} not considered"
    cur_joint_positions = PU.get_joint_positions(self.robot_id,joints=self.arm_ids)

    custom_limits = {}
    if joint_positions is None:
      if use_ikfast:
        if link_id==self.ee_id:
          ee_in_world = link_pose.copy()
          base_in_world = get_ob_pose_in_world(self.robot_id)
          ee_in_base = np.linalg.inv(base_in_world)@ee_in_world
        elif link_id==self.gripper_id:
          base_in_world = get_ob_pose_in_world(self.robot_id)
          gripper_in_base = np.linalg.inv(base_in_world)@link_pose
          ee_in_base = gripper_in_base@np.linalg.inv(self.gripper_in_ee)
        else:
          raise NameError
        if check_end_collision:
          end_obstacles = obstacles
        else:
          end_obstacles = []
        sols = self.ik_fast_feasible_solutions(ee_in_base,obstacles=end_obstacles,attachments=attachments)
        if len(sols)==0:
          print('[WARN] move_arm failed: feasible sol is None by ikfast')
          return None
        assert len(sols[0])==len(self.arm_ids), f'{len(sols[0])}!={len(self.arm_ids)}'
      else:
        joint_positions = PU.inverse_kinematics(self.robot_id,link_id,target_pose=link_pose,ignore_joint_limits=ignore_joint_limits,custom_limits=custom_limits)
        if joint_positions is None:
          print('[WARN] move_arm failed: joint_positions is None')
          return None
        joint_positions = joint_positions[:len(self.arm_ids)]

    if teleport:
      if use_ikfast:
        joint_positions = sols[0]
      PU.set_joint_positions(self.robot_id, joints=self.arm_ids, values=joint_positions)
      return 1


    if use_ikfast:
      for sol in sols:
        command = self.path_plan(sol,obstacles,resolutions=resolutions,attachments=attachments)
        if command is not None:
          return command
      print('[WARN] move_arm failed: ikfast path_plan failed')
      return None
    else:
      return self.path_plan(joint_positions,obstacles,resolutions=resolutions,attachments=attachments)

  def path_plan(self,joint_positions,obstacles,conf0=None,ignore_all_collsion=False,custom_limits={},attachments=[],collision_fn=None,resolutions=None,check_end_collision=True):
    conf0 = kuka_primitives.BodyConf(self.robot_id,joints=self.arm_ids)
    conf1 = kuka_primitives.BodyConf(self.robot_id,joints=self.arm_ids,configuration=joint_positions)
    free_motion_fn = kuka_primitives.get_free_motion_gen(self.robot_id, fixed=obstacles, teleport=False, self_collisions=False, ignore_all_collsion=ignore_all_collsion,custom_limits=custom_limits,attachments=attachments,collision_fn=collision_fn,resolutions=resolutions,check_end_collision=check_end_collision)
    res = free_motion_fn(conf0, conf1)
    if res is None:
      return None
    command = res[0]
    return command


  def move_arm_catesian(self,link_id,end_pose,move_step=0.001,timeout=999999,attachments=[],obstacles=[],custom_limits={},collision_fn=None,check_end_collision=True):
    assert link_id in [self.ee_id, self.gripper_id], f'link_id {link_id} wrong'
    state_id = p.saveState()
    if link_id==self.gripper_id:
      gripper_in_ee = get_pose_A_in_B(self.robot_id,self.gripper_id,self.robot_id,self.ee_id)
      tmp = np.eye(4)
      target_ee_in_world = end_pose@np.linalg.inv(gripper_in_ee)
      end_pos = target_ee_in_world[:3,3]
    else:
      end_pos = end_pose[:3,3]

    waypoint_poses = []
    ee_in_world = get_link_pose_in_world(self.robot_id,self.ee_id)
    start_pos = ee_in_world[:3,3]
    print(f'start_pos={start_pos}, end_pos={end_pos}')
    dir = end_pos-start_pos
    dir = dir/np.linalg.norm(dir)
    dist = np.linalg.norm(end_pos-start_pos)

    for travel in np.arange(0,dist+move_step,move_step):
      travel = min(travel,dist)
      trans =  start_pos + dir*travel
      q_wxyz = quaternion_from_matrix(ee_in_world)
      q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
      waypoint_poses.append((trans,q_xyzw))

    joint_positionss = PU.plan_cartesian_motion_ikfast(self.robot_id, arm_joints=self.arm_ids, ee_link=self.ee_id, waypoint_poses=waypoint_poses, get_ik_func=self.get_ik_func, custom_limits=custom_limits,obstacles=obstacles,attachments=attachments,collision_fn=collision_fn,check_end_collision=check_end_collision)

    if joint_positionss is None:
      print('[WARN] move_arm_catesian::joint_positionss is None')
      p.restoreState(state_id)
      p.removeState(state_id)
      return None
    arm_joint_positionss = []
    for joint_positions in joint_positionss:
      joint_positions = joint_positions[:len(self.arm_ids)]
      arm_joint_positionss.append(joint_positions)
    body_path = kuka_primitives.BodyPath(self.robot_id,arm_joint_positionss,joints=self.arm_ids,attachments=attachments)
    command = kuka_primitives.Command([body_path])
    p.restoreState(state_id)
    p.removeState(state_id)
    return command


  def restore_from_meta_file(self,meta_dir):
    with open(meta_dir,'rb') as ff:
      meta = pickle.load(ff)
    id_to_obj_file = meta['id_to_obj_file']

    self.cam_in_world = meta['cam_in_world']

    ob_in_worlds = []
    for body_id,ob_in_world in meta['poses'].items():
      if body_id in meta['env_body_ids']:
        continue
      ob_in_worlds.append(ob_in_world)
      obj_file = meta['id_to_obj_file'][body_id]
      scale = meta['id_to_scales'][body_id]

    print('obj_file',obj_file)
    print('scale',scale)

    ob_ids = create_duplicate_object(len(ob_in_worlds),obj_file,scale,ob_in_worlds,mass=0.1,has_collision=True,concave=False)
    for ob_id in ob_ids:
      self.id_to_obj_file[ob_id] = obj_file
      self.id_to_scales[ob_id] = scale
      p.changeDynamics(ob_id,-1,linearDamping=0.9,angularDamping=0.9,lateralFriction=0.9,spinningFriction=0.9)

    print("Restored from {}".format(meta_dir))


  def add_duplicate_object_on_pile(self,obj_file,scale,n_ob):
    '''
    @scale: (3) array
    '''
    ob_in_worlds = []
    bin_pose = get_ob_pose_in_world(self.bin_id)
    for i in range(n_ob):
      ob_x = np.random.uniform(-self.bin_dimensions[0]/2,self.bin_dimensions[0]/2) + bin_pose[0,3]
      ob_y = np.random.uniform(-self.bin_dimensions[1]/2,self.bin_dimensions[1]/2) + bin_pose[1,3]
      ob_z = np.random.uniform(0.05,1) + bin_pose[2,3]
      ob_pos = np.array([ob_x,ob_y,ob_z])
      R = random_rotation_matrix(np.random.rand(3))
      q_wxyz = quaternion_from_matrix(R)
      q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
      ob_in_world = np.eye(4)
      ob_in_world[:3,3] = ob_pos
      ob_in_world[:3,:3] = R[:3,:3]
      ob_in_worlds.append(ob_in_world)

    ob_ids = create_duplicate_object(n_ob,obj_file,scale,ob_in_worlds,mass=0.1,has_collision=True,concave=False)
    for ob_id in ob_ids:
      self.id_to_obj_file[ob_id] = copy.deepcopy(obj_file)
      self.id_to_scales[ob_id] = scale.copy()
      p.changeDynamics(ob_id,-1,linearDamping=0.9,angularDamping=0.9,lateralFriction=0.9,spinningFriction=0.9,collisionMargin=0.0001)
    return ob_ids


  def simulation_until_stable(self):
    print('simulation_until_stable....')
    n_step = 0
    while 1:
      bin_in_world = get_ob_pose_in_world(self.bin_id)
      for body_id in PU.get_bodies():
        if body_id in self.env_body_ids:
          continue
        ob_in_world = get_ob_pose_in_world(body_id)
        ob_in_bin = np.linalg.inv(bin_in_world)@ob_in_world
        if ob_in_bin[2,3]<=-0.02 or np.abs(ob_in_bin[0,3])>0.05 or np.abs(ob_in_bin[1,3])>0.05:  # Out of bin
          p.removeBody(body_id)

      last_poses = {}
      accum_motions = {}
      for body_id in PU.get_bodies():
        if body_id in self.env_body_ids:
          continue
        last_poses[body_id] = get_ob_pose_in_world(body_id)
        accum_motions[body_id] = 0

      stabled = True
      for _ in range(50):
        p.stepSimulation()
        n_step += 1
        for body_id in PU.get_bodies():
          if body_id in self.env_body_ids:
            continue
          cur_pose = get_ob_pose_in_world(body_id)
          motion = np.linalg.norm(cur_pose[:3,3]-last_poses[body_id][:3,3])
          accum_motions[body_id] += motion
          last_poses[body_id] = cur_pose.copy()
          if accum_motions[body_id]>=0.001:
            stabled = False
            break
        if stabled==False:
          break

      if stabled:
        for body_id in PU.get_bodies():
          if body_id in self.env_body_ids:
            continue
          p.resetBaseVelocity(body_id,linearVelocity=[0,0,0],angularVelocity=[0,0,0])
        break

    print('Finished simulation')


  def make_pile(self,obj_file,scale_range,n_ob_range,remove_condition=None):
    scale = np.random.uniform(scale_range[0],scale_range[1])
    mesh = trimesh.load(obj_file)
    dimension = mesh.vertices.max(axis=0)-mesh.vertices.min(axis=0)
    min_scale = 0.005/dimension.min()  # Shortest side larger than 0.005
    max_scale = 0.1/dimension.max()
    scale = np.clip(scale,min_scale,max_scale)
    scale = np.array([scale,scale,scale])

    print('Making pile {} scale={}'.format(obj_file,scale))

    body_ids = PU.get_bodies()
    for body_id in body_ids:
      p.changeDynamics(body_id,-1,activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)

    num_objects = np.random.randint(n_ob_range[0],n_ob_range[1])
    print("Add new objects on pile #={}".format(num_objects))

    before_ids = PU.get_bodies()

    while 1:
      new_ids = list(set(PU.get_bodies())-set(before_ids))
      if remove_condition is not None:
        for id in new_ids:
          if remove_condition(id):
            p.removeBody(id)

      new_ids = list(set(PU.get_bodies())-set(before_ids))
      if len(new_ids)==num_objects:
        break
      if len(new_ids)>num_objects:
        to_remove_ids = np.random.choice(np.array(new_ids),size=len(new_ids)-num_objects,replace=False)
        for id in to_remove_ids:
          p.removeBody(id)
        self.simulation_until_stable()
        continue
      self.add_duplicate_object_on_pile(obj_file,scale,num_objects-len(new_ids))
      self.simulation_until_stable()


    self.ob_ids = list(set(PU.get_bodies())-set(before_ids))



  def generate_one(self,obj_file,data_id,scale_range,n_ob_range):
    begin = time.time()

    self.make_pile(obj_file=obj_file,scale_range=scale_range,n_ob_range=n_ob_range)

    n_trial = 0
    while 1:
      n_trial += 1
      if n_trial>=5:
        self.reset()
        self.generate_one(obj_file,data_id,scale_range,n_ob_range)
        return

      bin_pose = get_ob_pose_in_world(self.bin_id)
      self.cam_in_world = bin_pose@self.cam_in_bin

      rgb,depth,seg = self.camera.render(self.cam_in_world)
      seg[seg<0] = 0
      if seg.max()>=65535:
        raise RuntimeError('seg.max={} reaches uint16 limit'.format(seg.max()))

      seg_ids = np.unique(seg)
      if len(set(seg_ids)-set(self.env_body_ids))==0:
        print(f'Need continue. seg_ids={seg_ids}, self.env_body_ids={self.env_body_ids}')
        continue

      break


    rgb_dir = '{}/{:07d}rgb.png'.format(self.out_dir,data_id)
    Image.fromarray(rgb).save(rgb_dir)
    cv2.imwrite(rgb_dir.replace('rgb','depth'), (depth*10000).astype(np.uint16))
    cv2.imwrite(rgb_dir.replace('rgb','seg'), seg.astype(np.uint16))
    poses = {}
    for body_id in PU.get_bodies():
      if body_id in self.env_body_ids:
        continue
      poses[body_id] = get_ob_pose_in_world(body_id)

    with open(rgb_dir.replace('rgb.png','meta.pkl'),'wb') as ff:
      meta = {'cam_in_world':self.cam_in_world, 'K':self.K, 'id_to_obj_file':self.id_to_obj_file, 'poses':poses, 'id_to_scales':self.id_to_scales, 'env_body_ids':self.env_body_ids}
      pickle.dump(meta,ff)
    print("Saved to {}".format(rgb_dir))

    print("Generate one sample time {} s".format(time.time()-begin))
    print(">>>>>>>>>>>>>>>>>>>>>>")
