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



def compute_nunocs_label_worker(color_file):
  color = np.array(Image.open(color_file))
  depth = cv2.imread(color_file.replace('rgb','depth'),-1)/1e4
  H,W = depth.shape[:2]
  with open(color_file.replace('rgb.png','meta.pkl'),'rb') as ff:
    meta = pickle.load(ff)

  code_dir = os.path.dirname(os.path.realpath(__file__))
  with open(f'{code_dir}/config.yml','r') as ff:
    cfg = yaml.safe_load(ff)

  mesh_pts = None
  env_body_ids = meta['env_body_ids']
  K = meta['K']
  poses = meta['poses']
  xyz_map = depth2xyzmap(depth,K)
  seg = cv2.imread(color_file.replace('rgb','seg'),-1)
  seg_ids = np.unique(seg)
  nocs_image = np.zeros((H,W,3))
  for seg_id in seg_ids:
    if seg_id in env_body_ids:
      continue
    if np.sum(seg==seg_id)==0:
      continue

    if mesh_pts is None:
      mesh_dir = meta['id_to_obj_file'][seg_id]
      class_name = get_class_name(mesh_dir)
      symmetry_tfs = get_symmetry_tfs(class_name,allow_reflection=False)
      mesh_scale = meta['id_to_scales'][seg_id]
      mesh = trimesh.load(mesh_dir)
      mesh_pts = mesh.vertices.copy()*mesh_scale
      max_xyz = mesh_pts.max(axis=0).reshape(1,3)
      min_xyz = mesh_pts.min(axis=0).reshape(1,3)
      center_xyz = (max_xyz+min_xyz)/2

    valid_mask = (seg==seg_id) & (xyz_map[...,2]>=0.1)
    tmp_xyz = xyz_map[valid_mask].reshape(-1,3)
    ob_in_world = poses[seg_id].copy()
    cam_in_world = meta['cam_in_world'].copy()
    ob_in_cam = np.linalg.inv(cam_in_world)@ob_in_world
    cam_in_ob = np.linalg.inv(ob_in_cam)
    tmp_xyz = (cam_in_ob@to_homo(tmp_xyz).T).T[:,:3]

    nunocs_scale = 1.
    nocs_xyz = (tmp_xyz-center_xyz) / (max_xyz-min_xyz).reshape(1,3)  #[-0.5,0.5]
    nocs_xyz = np.clip(nocs_xyz,-0.5,0.5)
    nocs_xyz /= nunocs_scale
    nocs_image[valid_mask] = (nocs_xyz+0.5)*255

  out_file = color_file.replace('rgb','nunocs')
  nocs_image = np.clip(nocs_image,0,255)
  nocs_image = nocs_image.round().astype(np.uint8)
  Image.fromarray(nocs_image).save(out_file)
  print(f"Write to {out_file}")


def compute_nunocs_label():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  color_files = sorted(glob.glob(f'{code_dir}/dataset/{class_name}/**/*rgb.png',recursive=True))
  print("#color_files={}".format(len(color_files)))

  for color_file in color_files:
    compute_nunocs_label_worker(color_file)


def fill_depth_normal_worker(reader,depth_file):
  data = reader.read_data_by_colorfile(depth_file.replace('depth.png','rgb.png'),fetch=['xyz_map'])
  valid_mask = data['xyz_map'][:,:,2]>=0.1
  pts = data['xyz_map'][valid_mask].reshape(-1,3)
  pcd = toOpen3dCloud(pts)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
  pcd = correct_pcd_normal_direction(pcd)
  normals = np.asarray(pcd.normals).copy()
  normal_map = np.zeros(data['xyz_map'].shape)
  vs,us = np.where(valid_mask>0)
  normal_map[vs,us] = normals
  normal_map = np.round((normal_map+1)/2.0*255)
  normal_map = np.clip(normal_map,0,255).astype(np.uint8)
  out_file = depth_file.replace('depth','normal')
  Image.fromarray(normal_map).save(out_file)
  print(f"Write to {out_file}")



def fill_depth_normal():
  reader = DataReader(cfg)
  code_dir = os.path.dirname(os.path.realpath(__file__))
  depth_files = sorted(glob.glob(f'{code_dir}/dataset/{class_name}/**/*depth.png',recursive=True))
  print("#depth_files={}".format(len(depth_files)))

  for depth_file in depth_files:
    fill_depth_normal_worker(reader,depth_file)


def make_isolated_training_data_worker(cfg,color_file):
  '''
  Isolate objects in the scene
  '''
  print('color_file',color_file)
  reader = DataReader(cfg)
  data = reader.read_data_by_colorfile(color_file)
  seg_ids = np.unique(data['seg'])
  seg_ids = seg_ids[seg_ids>0]

  for seg_id in seg_ids:
    mask = (data['seg']==seg_id) & (data['depth']>=0.1)
    if np.sum(mask)<100:
      continue

    cloud_xyz = data['xyz_map'][mask].reshape(-1,3)
    cloud_nocs = data['nocs_map'][mask].reshape(-1,3)
    cloud_rgb = data['rgb'][mask].reshape(-1,3)
    cloud_normal = data['normal_map'][mask].reshape(-1,3)

    out_data = {'cloud_xyz':cloud_xyz, 'cloud_nocs':cloud_nocs, 'cloud_rgb':cloud_rgb, 'cloud_normal':cloud_normal, 'color_file':color_file, 'seg_id':seg_id}
    out_dir = color_file.replace('/train/','/train_isolated_nunocs/').replace('/test/','/test_isolated_nunocs/').replace('rgb.png','_seg{}.pkl'.format(seg_id))
    os.system("mkdir -p {}".format(os.path.dirname(out_dir)))
    with gzip.open(out_dir,'wb') as ff:
      pickle.dump(out_data,ff)


def make_isolated_training_data():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  color_files = sorted(glob.glob(f'{code_dir}/dataset/{class_name}/**/*rgb.png',recursive=True))

  for color_file in color_files:
    make_isolated_training_data_worker(cfg,color_file)



def make_crop_scene_dataset_worker(color_file,out_dir,reader,n_crop_per_side,downsample_size):
  print(color_file)
  index_str = re.findall(r'[0-9]{7}',color_file)[0]
  data = reader.read_data_by_colorfile(color_file,fetch=['xyz_map','normal_map'])
  valid_mask = data['depth']>=0.1

  data = {
    'cloud_xyz': data['xyz_map'][valid_mask].reshape(-1,3),
    'cloud_rgb': data['rgb'][valid_mask].reshape(-1,3),
    'cloud_normal': data['normal_map'][valid_mask].reshape(-1,3),
    'cloud_seg': data['seg'][valid_mask].reshape(-1),
  }

  pcd = toOpen3dCloud(data['cloud_xyz'])
  pcd = pcd.voxel_down_sample(voxel_size=downsample_size)
  kdtree = cKDTree(np.asarray(pcd.points))
  indices,dists = kdtree.query(data['cloud_xyz'])

  max_xyz = data['cloud_xyz'].max(axis=0)
  min_xyz = data['cloud_xyz'].min(axis=0)
  crop_x_length = (max_xyz[0]-min_xyz[0])/n_crop_per_side
  crop_y_length = (max_xyz[1]-min_xyz[1])/n_crop_per_side
  print(f'crop_x_length',crop_x_length)

  for x_crop in range(n_crop_per_side):
    for y_crop in range(n_crop_per_side):
      xmin = min_xyz[0]+x_crop*crop_x_length
      xmax = xmin+crop_x_length+0.001
      ymin = min_xyz[1]+y_crop*crop_y_length
      ymax = ymin+crop_y_length+0.001
      keep_mask = (data['cloud_xyz'][:,0]>=xmin) & (data['cloud_xyz'][:,0]<=xmax) & \
                  (data['cloud_xyz'][:,1]>=ymin) & (data['cloud_xyz'][:,1]<=ymax)
      cropped_data = {}
      for k in data.keys():
        cropped_data[k] = data[k][keep_mask]

      out_file = f'{out_dir}/{index_str}_x_crop_{x_crop}_y_crop_{y_crop}.pkl'
      with gzip.open(out_file,'wb') as ff:
        pickle.dump(cropped_data,ff)

      if int(index_str)<5:
        pcd = toOpen3dCloud(cropped_data['cloud_xyz'],cropped_data['cloud_rgb'],cropped_data['cloud_normal'])
        o3d.io.write_point_cloud(out_file.replace('.pkl','.ply'),pcd)


def make_crop_scene_dataset():
  '''For instance segmentation training, remove background e.g. bin
  '''
  reader = DataReader(cfg)
  code_dir = os.path.dirname(os.path.realpath(__file__))
  base_dir = f'{code_dir}/dataset/{class_name}'
  for split in ['train','test']:
    color_files = sorted(glob.glob(f'{base_dir}/{split}/*rgb.png',recursive=True))
    print(f'color_files={len(color_files)}')

    n_crop_per_side = 1
    downsample_size = 0.0005

    out_dir = f'{base_dir}/{split}_cloud_n_crop_per_side_{n_crop_per_side}_downsample_size_{round(downsample_size,5)}'
    print(f'out_dir: {out_dir}')
    os.system(f'rm -rf {out_dir} && mkdir -p {out_dir}')

    for color_file in color_files:
      make_crop_scene_dataset_worker(color_file,out_dir,reader,n_crop_per_side,downsample_size)




def compute_per_ob_visibility_worker(color_file,cfg):
  out_file = color_file.replace('rgb.png','full_vis_mask.pkl')
  if os.path.exists(out_file):
    return
  with open(color_file.replace('rgb.png','meta.pkl'),'rb') as ff:
    meta = pickle.load(ff)
  seg = cv2.imread(color_file.replace('rgb','seg'),-1)
  seg_ids = np.unique(seg)
  id_to_obj_file = meta['id_to_obj_file']
  model_dir = None
  for body_id in seg_ids:
    if body_id in meta['env_body_ids']:
      continue
    if model_dir is None:
      obj_file = meta['id_to_obj_file'][body_id]
      mesh = trimesh.load(obj_file)
      scale = meta['id_to_scales'][body_id]
      mesh.vertices = mesh.vertices*scale.reshape(1,3)
      model_dir = '/tmp/{}.obj'.format(os.path.basename(color_file).replace('.png',''))
      mesh.export(model_dir)
      break
  K = np.array(meta['K']).reshape(3,3)
  renderer = ModelRendererOffscreen([model_dir],K,H=cfg['H'],W=cfg['W'])
  cam_in_world = meta['cam_in_world']

  full_vis_mask = {}
  for seg_id in seg_ids:
    if seg_id in meta['env_body_ids']:
      continue
    ob_in_world = meta['poses'][seg_id]
    ob_in_cam = np.linalg.inv(cam_in_world)@ob_in_world
    color,depth = renderer.render([ob_in_cam])
    vs,us = np.where(depth>=0.1)
    full_vis_mask[seg_id] = np.concatenate((us.reshape(-1,1),vs.reshape(-1,1)),axis=1).astype(np.uint16)
  with gzip.open(out_file,'wb') as ff:
    pickle.dump(full_vis_mask,ff)

  os.remove(model_dir)


def compute_per_ob_visibility():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  color_files = sorted(glob.glob(f'{code_dir}/dataset/{class_name}/**/*rgb.png',recursive=True))
  print('#color_files={}'.format(len(color_files)))

  for color_file in color_files:
    compute_per_ob_visibility_worker(color_file,cfg)




def make_dense_clutter_grasp_data_worker(color_file,cfg,gripper,grasps):
  print('color_file',color_file)
  with open(color_file.replace('rgb.png','meta.pkl'),'rb') as ff:
    meta = pickle.load(ff)
  K = np.array(meta['K']).reshape(3,3)

  with gzip.open(color_file.replace('rgb.png','full_vis_mask.pkl'),'rb') as ff:
    full_vis_mask = pickle.load(ff)

  id_to_obj_file = meta['id_to_obj_file']
  cam_in_world = meta['cam_in_world']
  for body_id,obj_file in id_to_obj_file.items():
    if body_id in meta['env_body_ids']:
      continue
    obj_file = id_to_obj_file[body_id]
    scales = meta['id_to_scales'][body_id]
    scales_tf = np.eye(4)
    scales_tf[:3,:3] = np.diag(scales)
    break

  grasp_in_gripper = gripper.get_grasp_pose_in_gripper_base()
  depth = cv2.imread(color_file.replace('rgb','depth'),-1)/1e4
  xyz_map = depth2xyzmap(depth,K)
  scene_pts = xyz_map[xyz_map[:,:,2]>=0.1].reshape(-1,3)
  pcd = toOpen3dCloud(scene_pts)
  pcd = pcd.voxel_down_sample(voxel_size=0.001)
  scene_pts = np.asarray(pcd.points).copy()
  seg = cv2.imread(color_file.replace('rgb','seg'),-1)

  out_grasps = {}
  check_finger_region = False
  max_grasp_per_scene = 20
  candidates = []
  body_ids = np.unique(seg)
  np.random.shuffle(body_ids)
  n_rej = {
    'dot': 0,
    'gripper_region': 0,
    'collision_with_scene': 0,
  }
  candidate_grasps = []
  vis_ratio_dict = {}
  for body_id in body_ids:
    if body_id in meta['env_body_ids']:
      continue
    n_visible = (seg==body_id).sum()
    n_full_vis = full_vis_mask[body_id].shape[0]
    vis_ratio = n_visible/n_full_vis
    vis_ratio_dict[body_id] = vis_ratio
    ob_pts = xyz_map[seg==body_id].reshape(-1,3)
    ob_pts = ob_pts[ob_pts[:,2]>=0.1]
    pcd = toOpen3dCloud(ob_pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    ob_pts = np.asarray(pcd.points)
    if vis_ratio>=0.8:
      grasp_ids = np.arange(len(grasps))
      np.random.shuffle(grasp_ids)
      for grasp_id in grasp_ids:
        grasp = grasps[grasp_id]
        grasp_pose = grasp.grasp_pose
        grasp_pose = scales_tf@grasp_pose
        grasp_pose = normalizeRotation(grasp_pose)
        ob_in_world = meta['poses'][body_id]
        ob_in_cam = np.linalg.inv(cam_in_world)@ob_in_world
        grasp_in_cam = ob_in_cam@grasp_pose
        approach_dir = (grasp_in_cam[:3,:3]@np.array([1,0,0]).reshape(3,1)).reshape(3)
        dot = np.dot(approach_dir,np.array([0,0,1]))
        if dot<0:
          n_rej['dot'] += 1
          continue

        if check_finger_region:
          ob_in_grasp = (np.linalg.inv(grasp_in_cam)@to_homo(ob_pts).T).T[:,:3]
          valid_mask = (ob_in_grasp[:,0]>=gripper.finger_xmin) & (ob_in_grasp[:,0]<=gripper.finger_xmax) & (ob_in_grasp[:,1]>=gripper.finger_ymin) & (ob_in_grasp[:,1]<=gripper.finger_ymax) & (ob_in_grasp[:,2]>=gripper.finger_zmin) & (ob_in_grasp[:,2]<=gripper.finger_zmax)

          if valid_mask.sum()==0:
            n_rej['gripper_region'] += 1
            continue

        candidate_grasps.append([body_id,grasp_in_cam,grasp.perturbation_score])
        continue

  candidate_grasps = np.array(candidate_grasps)
  print("candidate_grasps={}".format(len(candidate_grasps)))

  ids = np.random.choice(len(candidate_grasps),size=min(max_grasp_per_scene,len(candidate_grasps)),replace=False)
  candidate_grasps = candidate_grasps[ids]
  out_grasps = {}
  for (body_id,grasp_in_cam,grasp.perturbation_score) in candidate_grasps:
    if body_id not in out_grasps:
      out_grasps[body_id] = []
    out_grasps[body_id].append([grasp_in_cam,grasp.perturbation_score])

  msg = 'n_rej: '
  for k,n in n_rej.items():
    msg += f'{k}={n} '
  print(msg)

  out_file = color_file.replace('rgb.png','grasp.pkl')
  with gzip.open(out_file,'wb') as ff:
    pickle.dump(out_grasps,ff)



def make_dense_clutter_grasp_data():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  color_files = sorted(glob.glob(f'{code_dir}/dataset/{class_name}/**/*rgb.png',recursive=True))
  print("#color_files={}".format(len(color_files)))

  names = cfg['dataset'][class_name]['train']
  grasps_dict = {}
  for name in names:
    code_dir = os.path.dirname(os.path.realpath(__file__))
    with gzip.open(f"{code_dir}/data/object_models/{name.replace('.obj','_grasp_balanced_score.pkl')}",'rb') as ff:
      grasps = pickle.load(ff)
      grasps_dict[name] = grasps

  grasps_all = []
  for i,color_file in enumerate(color_files):
    if i%max(len(color_files)//10,1)==0:
      print(f"Preparing grasps for each scene {i}/{len(color_files)}")
    seg = cv2.imread(color_file.replace('rgb','seg'),-1)
    seg_ids = np.unique(seg)
    with open(color_file.replace('rgb.png','meta.pkl'),'rb') as ff:
      meta = pickle.load(ff)
    for body_id in seg_ids:
      if body_id in meta['env_body_ids']:
        continue
      obj_file = meta['id_to_obj_file'][body_id]
      ob_name = os.path.basename(obj_file)
      break
    grasps = np.array(grasps_dict[ob_name])
    grasps = np.random.choice(grasps,size=min(100,len(grasps)),replace=False)
    grasps_all.append(grasps)

  del grasps_dict

  for i in range(len(color_files)):
    make_dense_clutter_grasp_data_worker(color_files[i],cfg,gripper,grasps_all[i])


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

  compute_nunocs_label()
  fill_depth_normal()
  compute_per_ob_visibility()
  make_isolated_training_data()
  make_crop_scene_dataset()
  make_dense_clutter_grasp_data()
