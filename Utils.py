import open3d as o3d
import os, sys, time,torch,pickle,trimesh,yaml
from scipy.spatial import ConvexHull
from uuid import uuid4
import cv2
from PIL import Image, ImageDraw
import numpy as np
import multiprocessing as mp
import math,glob,re,copy
from transformations import *
from scipy.spatial import cKDTree
from collections import OrderedDict


place_pose_dict = {}  # Placement pose relative to placeholder, a pair of pose before and after place
place_pose_dict['nut'] = [np.eye(4),np.eye(4)]
place_pose_dict['nut'][0][:3,3] += np.array([0,0,0.15])
place_pose_dict['nut'][1][:3,3] += np.array([0,0,0.08])
place_pose_dict['hnm'] = [np.eye(4),np.eye(4)]
place_pose_dict['hnm'][0][:3,3] -= np.array([0,0,0.05])
place_pose_dict['hnm'][1][:3,3] -= np.array([0,0,0.02])
place_pose_dict['screw'] = [np.eye(4),np.eye(4)]
place_pose_dict['screw'][0][:3,3] = [0,0,-0.07]
place_pose_dict['screw'][1][:3,3] = [0,0,-0.02]


def get_class_name(ob_dir):
  if '/nut' in ob_dir:
    return 'nut'
  elif '/hnm' in ob_dir:
    return 'hnm'
  elif '/screw' in ob_dir:
    return 'screw'
  else:
    raise RuntimeError(f'class name not found {ob_dir}')


def get_place_success_func(class_name):
  if class_name=='nut':
    def func(ob_pose,place_pose):
      if np.linalg.norm(ob_pose[:2,3]-place_pose[:2,3])>0.005:
        print('placement check failed: center dist',np.linalg.norm(ob_pose[:2,3]-place_pose[:2,3]))
        return False
      if np.abs(ob_pose[2,3]-place_pose[2,3])>0.02:
        print(f'placement check failed: height wrong, ob_pose[2,3]={ob_pose[2,3]}, place_pose[2,3]={place_pose[2,3]}')
        return False
      return True

  elif class_name=='hnm':
    def func(ob_pose,place_pose):
      if np.linalg.norm(ob_pose[:2,3]-place_pose[:2,3])>0.005:
        print('placement check failed: center dist',np.linalg.norm(ob_pose[:2,3]-place_pose[:2,3]))
        return False
      ob_dir = (ob_pose[:3,:3]@np.array([0,0,1]).reshape(3,1)).reshape(3)
      place_dir = (place_pose[:3,:3]@np.array([0,0,1]).reshape(3,1)).reshape(3)
      dot = np.dot(ob_dir,place_dir)
      if np.abs(dot)<np.cos(80/180.0*np.pi):
        print('placement failed: orientation not parallel')
        return False
      return True

  elif class_name=='screw':
    def func(ob_pose,place_pose):
      xy_dist = np.linalg.norm(ob_pose[:2,3]-place_pose[:2,3])
      if xy_dist>=0.01:
        print('placement check failed: center dist',xy_dist)
        return False
      ob_dir = (ob_pose[:3,:3]@np.array([0,0,1]).reshape(3,1)).reshape(3)
      place_dir = (place_pose[:3,:3]@np.array([0,0,1]).reshape(3,1)).reshape(3)
      dot = np.dot(ob_dir,place_dir)
      if np.abs(dot)<np.cos(80/180.0*np.pi):
        print('placement failed: orientation not parallel')
        return False
      return True

  return func


def get_symmetry_tfs(class_name,allow_reflection=True):
  tfs = []
  if class_name=='nut':
    for xangle in np.arange(0,360,180)/180*np.pi:
      for zangle in np.arange(0,360,60)/180*np.pi:
        tf = euler_matrix(xangle,0,zangle,axes='sxyz')
        tfs.append(tf)

  elif class_name=='hnm':
    for rz in [0,np.pi]:
      tf = euler_matrix(0,0,rz,axes='sxyz')
      tfs.append(tf)
  elif class_name=='screw':
    for zrot in np.arange(0,360,5)/180.0*np.pi:
      tf = euler_matrix(0,0,zrot,axes='sxyz')
      tfs.append(tf)
  else:
    raise RuntimeError(f'{class_name} not found')

  if not allow_reflection:
    new_tfs = []
    for i in range(len(tfs)):
      if np.linalg.det(tfs[i][:3,:3])<0:
        continue
      new_tfs.append(tfs[i])
    tfs = new_tfs

  return np.array(tfs)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



def is_pose_matrix_close(poseA,poseB,trans_tol,rot_tol,verbose=False):
  '''
  @trans_tol: in meter
  @rot_tol: angle in deg
  '''
  trans_err = np.linalg.norm(poseA[:3,3]-poseB[:3,3])
  if verbose:
      print('trans err:',trans_err)

  if trans_err>=trans_tol:
    return False

  angle = geodesic_distance(poseA[:3,:3],poseB[:3,:3])
  if verbose:
      print('rot err deg:',angle/np.pi*180)

  if np.abs(angle)>=rot_tol/180.0*np.pi:
    return False

  return True


def load_model(model,ckpt_dir):
  state_dict = torch.load(ckpt_dir,map_location=torch.device("cpu"))
  if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
  print("Load ckpt from {}".format(ckpt_dir))

  try:
    new_state_dict = OrderedDict()
    for name in state_dict.keys():
      new_state_dict[name.replace('module.','')] = state_dict[name]
    state_dict = new_state_dict
    assert len(state_dict)>0
    model.load_state_dict(state_dict)
    return model
  except Exception as e:
    print(e)
    print('*'*100)
    print("Current model layers:")
    cur_layers = []
    for name,param in model.named_parameters():
        print(name)
        cur_layers.append(name)
    print('*'*100)
    print("ckpt layers:")
    ckpt_layers = []
    for name,param in state_dict.items():
        print(name)
        ckpt_layers.append(name)
    print('*'*100)
    print("Difference:")
    for layer in cur_layers:
        if layer not in ckpt_layers:
            print('{} not found in ckpt'.format(layer))
    for layer in ckpt_layers:
        if layer not in cur_layers:
            print('{} not found in cur model'.format(layer))
    raise RuntimeError

def normalizeRotation(pose):
  '''Assume no shear case
  '''
  new_pose = pose.copy()
  scales = np.linalg.norm(pose[:3,:3],axis=0)
  new_pose[:3,:3] /= scales.reshape(1,3)
  return new_pose


def read_normal_image(img_dir):
  normal = np.array(Image.open(img_dir))
  normal = normal/255.0 * 2 - 1
  valid_mask = np.linalg.norm(normal,axis=-1)>0.1
  normal = normal/(np.linalg.norm(normal,axis=-1)[:,:,None]+1e-15)
  normal[valid_mask==0] = 0
  return normal.astype(np.float32)



def toOpen3dCloud(points,colors=None,normals=None):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
      if colors.max()>1:
        colors = colors/255.0
      cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
      cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def correct_pcd_normal_direction(pcd, view_port=np.zeros((3),dtype=float)):
  view_dir = view_port.reshape(-1,3)-np.asarray(pcd.points)   #(N,3)
  view_dir = view_dir/np.linalg.norm(view_dir,axis=1).reshape(-1,1)
  normals = np.asarray(pcd.normals)/(np.linalg.norm(np.asarray(pcd.normals),axis=1)+1e-10).reshape(-1,1)
  dots = (view_dir*normals).sum(axis=1)
  indices = np.where(dots<0)
  normals[indices,:] = -normals[indices,:]
  pcd.normals = o3d.utility.Vector3dVector(normals)
  return pcd


def value_to_heatmap_rgb(minimum, maximum, value):
  minimum, maximum = float(minimum), float(maximum)
  ratio = 2 * (value-minimum) / (maximum - minimum)
  b = int(max(0, 255*(1 - ratio)))
  r = int(max(0, 255*(ratio - 1)))
  g = 255 - b - r
  return np.array([r, g, b])


def array_to_heatmap_rgb(a):
  '''
  @a: 1-d array
  '''
  minimum = a.min()
  maximum = a.max()
  ratio = 2 * (a-minimum) / (maximum - minimum)   # 0 to 2
  b = np.clip(255*(1 - ratio), 0, 255)
  r = np.clip(255*(ratio - 1), 0, 255)
  g = 255 - b - r
  return np.stack([r, g, b],axis=-1).reshape(-1,3).astype(np.uint8)  #(N,3)



def depth2xyzmap(depth, K):
	invalid_mask = (depth<0.1)
	H,W = depth.shape[:2]
	vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
	vs = vs.reshape(-1)
	us = us.reshape(-1)
	zs = depth.reshape(-1)
	xs = (us-K[0,2])*zs/K[0,0]
	ys = (vs-K[1,2])*zs/K[1,1]
	pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
	xyz_map = pts.reshape(H,W,3).astype(np.float32)
	xyz_map[invalid_mask] = 0
	return xyz_map.astype(np.float32)



def geodesic_distance(R1,R2):
  cos = (np.trace(R1.dot(R2.T))-1)/2
  cos = np.clip(cos,-1,1)
  return math.acos(cos)



def directionVecToRotation(direction, ref=np.array([0,0,1])):
  direction = direction/np.linalg.norm(direction)

  v = np.cross(direction,ref)
  if (v==0).all():
    R = np.eye(3)
    return R

  s = np.linalg.norm(v)

  c = direction.dot(ref)
  v_skew = [[0, -v[2], v[1]],
          [v[2], 0, -v[0]],
          [-v[1], v[0], 0]]
  v_skew = np.array(v_skew)

  if s==0: # opposite direction rotate around any axis
      R=[[1,0,0],
          [0,-1,0],
          [0,0,-1]]
      R = np.array(R)
  else:
      R = np.identity(3) + v_skew + v_skew.dot(v_skew)*(1-c)/(s**2) #from direction to ref
      R = R.T

  R = normalizeRotation(R)
  if np.linalg.norm(R.dot(ref)-direction)>1e-3:
    print("In directionVecToRotMat, rotation error {}".format(np.linalg.norm(R.dot(ref)-direction)))
  return R


def hinter_sampling(min_n_pts, radius=1):
  '''
  Sphere sampling based on refining icosahedron as described in:
  Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
  Local Patches with a Simple Linear Classifier, BMVC 2008

  :param min_n_pts: Minimum required number of points on the whole view sphere.
  :param radius: Radius of the view sphere.
  :return: 3D points on the sphere surface and a list that indicates on which
       refinement level the points were created.
  '''

  # Get vertices and faces of icosahedron
  a, b, c = 0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0
  pts = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
       (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b), (-c, a, b)]
  faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
       (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
       (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
       (8, 6, 7), (9, 8, 1)]

  # Refinement level on which the points were created
  pts_level = [0 for _ in range(len(pts))]

  ref_level = 0
  while len(pts) < min_n_pts:
    ref_level += 1
    edge_pt_map = {} # Mapping from an edge to a newly added point on that edge
    faces_new = [] # New set of faces

    # Each face is replaced by 4 new smaller faces
    for face in faces:
      pt_inds = list(face) # List of point IDs involved in the new faces
      for i in range(3):
        # Add a new point if this edge hasn't been processed yet,
        # or get ID of the already added point.
        edge = (face[i], face[(i + 1) % 3])
        edge = (min(edge), max(edge))
        if edge not in edge_pt_map.keys():
          pt_new_id = len(pts)
          edge_pt_map[edge] = pt_new_id
          pt_inds.append(pt_new_id)

          pt_new = 0.5 * (np.array(pts[edge[0]]) + np.array(pts[edge[1]]))
          pts.append(pt_new.tolist())
          pts_level.append(ref_level)
        else:
          pt_inds.append(edge_pt_map[edge])

      # Replace the current face with 4 new faces
      faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
              (pt_inds[3], pt_inds[1], pt_inds[4]),
              (pt_inds[3], pt_inds[4], pt_inds[5]),
              (pt_inds[5], pt_inds[4], pt_inds[2])]
    faces = faces_new

  # Project the points to a sphere
  pts = np.array(pts)
  pts *= np.reshape(radius / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))

  # Collect point connections
  pt_conns = {}
  for face in faces:
    for i in range(len(face)):
      pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
      pt_conns[face[i]].add(face[(i + 2) % len(face)])

  # Order the points - starting from the top one and adding the connected points
  # sorted by azimuth
  top_pt_id = np.argmax(pts[:, 2])
  pts_ordered = []
  pts_todo = [top_pt_id]
  pts_done = [False for _ in range(pts.shape[0])]

  def calc_azimuth(x, y):
    two_pi = 2.0 * math.pi
    return (math.atan2(y, x) + two_pi) % two_pi

  while len(pts_ordered) != pts.shape[0]:
    # Sort by azimuth
    pts_todo = sorted(pts_todo, key=lambda i: calc_azimuth(pts[i][0], pts[i][1]))
    pts_todo_new = []
    for pt_id in pts_todo:
      pts_ordered.append(pt_id)
      pts_done[pt_id] = True
      pts_todo_new += [i for i in pt_conns[pt_id]] # Find the connected points

    # Points to be processed in the next iteration
    pts_todo = [i for i in set(pts_todo_new) if not pts_done[i]]

  # Re-order the points and faces
  pts = pts[np.array(pts_ordered), :]
  pts_level = [pts_level[i] for i in pts_ordered]
  pts_order = np.zeros((pts.shape[0],))
  pts_order[np.array(pts_ordered)] = np.arange(pts.shape[0])
  for face_id in range(len(faces)):
    faces[face_id] = [pts_order[i] for i in faces[face_id]]

  return pts, pts_level




def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def to_homo_torch(pts):
  '''
  @pts: shape can be (B,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1)).to(pts.device).float()
  homo = torch.cat((pts, ones),dim=-1)
  return homo



def sph2cart(phi, theta, r):
    point_on_sphere = np.zeros(3)
    point_on_sphere[0] = r * math.sin(phi) * math.cos(theta)
    point_on_sphere[1] = r * math.sin(phi) * math.sin(theta)
    point_on_sphere[2] = r * math.cos(phi)
    return point_on_sphere


def random_direction(theta_range=[0,np.pi*2], phi_range=[0,np.pi]):
  # Random pose on a sphere : https://www.jasondavies.com/maps/random-points/
  theta = np.random.uniform(theta_range[0],theta_range[1])
  zmax = math.cos(phi_range[0])
  zmin = math.cos(phi_range[1])
  elev = np.random.uniform(zmin,zmax)
  phi = math.acos(elev)
  return sph2cart(phi, theta, 1)


def random_gaussian_magnitude(max_T, max_R):
  direction_T = random_direction()
  direction_T /= np.linalg.norm(direction_T)
  while 1:
    magn_T = np.random.normal(0,max_T)
    if abs(magn_T)<=max_T:
      break
  T = direction_T*magn_T
  direction_R = random_direction()
  direction_R = direction_R/np.linalg.norm(direction_R)
  while 1:
    magn_R = np.random.normal(0,max_R)  #degree
    if abs(magn_R)<=max_R:
      break
  rod = direction_R*magn_R/180.0*np.pi
  R = cv2.Rodrigues(rod)[0].reshape(3,3).copy()
  pose = np.eye(4)
  pose[:3,:3] = R
  pose[:3,3] = T.copy()
  return pose


def random_uniform_magnitude(max_T, max_R):
  '''
  @max_R: degree
  '''
  direction_T = random_direction()
  direction_T = direction_T/np.linalg.norm(direction_T)
  magn_T = np.random.uniform(0,max_T)
  T = direction_T*magn_T
  direction_R = random_direction()
  direction_R = direction_R/np.linalg.norm(direction_R)
  magn_R = np.random.uniform(0,max_R)
  rod = direction_R*magn_R/180.0*np.pi
  R = cv2.Rodrigues(rod)[0].reshape(3,3).copy()
  pose = np.eye(4)
  pose[:3,:3] = R
  pose[:3,3] = T.copy()
  return pose


def chamfer_distance_between_clouds_mutual(pts1,pts2):
  kdtree1 = cKDTree(pts1)
  dists1, indices1 = kdtree1.query(pts2)
  kdtree2 = cKDTree(pts2)
  dists2, indices2 = kdtree2.query(pts1)
  dists = np.concatenate([dists1,dists2],axis=0).reshape(-1)
  return dists

def cloudA_minus_cloudB(ptsA,ptsB,thres):
  kdtree = cKDTree(ptsA)
  indices_tuple = kdtree.query_ball_point(ptsB,r=thres,n_jobs=-1)
  remove_ids = np.unique(np.concatenate(indices_tuple,axis=0).reshape(-1)).astype(int)
  keep_ids = list(set(np.arange(len(ptsA)))-set(remove_ids))
  keep_ids = np.array(keep_ids).astype(int)
  return ptsA[keep_ids], keep_ids



def compute_cloud_resolution(pts,n_sample=100):
  ids = np.random.choice(len(pts),size=n_sample).astype(int)
  sample_pts = pts[ids]
  background_ids = np.array(list(set(np.arange(len(pts)))-set(ids))).astype(int)
  background_pts = pts[background_ids]
  kdtree = cKDTree(background_pts)
  dists,indices = kdtree.query(sample_pts)
  dists = np.array(dists[np.isfinite(dists)])
  resolution = np.sort(dists)[:10].mean()
  return resolution