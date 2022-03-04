import numpy as np
import open3d as o3d
from transformations import *
import os,sys,yaml,copy,pickle,time,cv2,socket,argparse,inspect,trimesh,operator,gzip,re,random,torch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from scipy.spatial import cKDTree
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append("{}/../".format(code_dir))
sys.path.append("{}/ss-pybullet".format(code_dir))
from dexnet.grasping.grasp import ParallelJawPtGrasp3D
from autolab_core import YamlConfig
from dexnet.grasping.grasp_sampler import PointConeGraspSampler,NocsTransferGraspSampler
from PIL import Image
from Utils import *
from data_reader import *
from pointnet2 import *
from aligning import *
import PointGroup.data.dataset_seg as dataset_seg
from PointGroup.model.pointgroup.pointgroup import PointGroup
import PointGroup.lib.pointgroup_ops.functions.pointgroup_ops as pointgroup_ops
import PointGroup.util.config as config_pg
import spconv
from spconv.modules import SparseModule
from dataset_nunocs import NunocsIsolatedDataset
from dataset_grasp import GraspDataset
from functools import partial
from sklearn.cluster import DBSCAN,MeanShift
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat



class GraspPredicter:
  def __init__(self,class_name):
    self.class_name_to_artifact_id = {
      'nut': 47,
      'hnm': 51,
      'screw': 50,
    }
    artifact_id = self.class_name_to_artifact_id[class_name]
    code_dir = os.path.dirname(os.path.realpath(__file__))
    artifact_dir = f"{code_dir}/artifacts/artifacts-{artifact_id}"
    print('GraspPredicter artifact_dir',artifact_dir)
    with open(f"{artifact_dir}/config_grasp.yml",'r') as ff:
      self.cfg = yaml.safe_load(ff)

    normalizer_dir = '{}/normalizer.pkl'.format(artifact_dir)
    if os.path.exists(normalizer_dir):
      with open(normalizer_dir,'rb') as ff:
        tmp = pickle.load(ff)
      self.cfg['mean'] = tmp['mean']
      self.cfg['std'] = tmp['std']

    self.dataset = GraspDataset(self.cfg,phase='test',class_name=class_name)

    self.model = PointNetCls(n_in=self.cfg['input_channel'],n_out=len(self.cfg['classes'])-1)
    self.model = load_model(self.model,ckpt_dir='{}/best_val.pth.tar'.format(artifact_dir))
    self.model.cuda().eval()


  def predict_batch(self,data,grasp_poses):
    with torch.no_grad():
      batch_size = 200
      input_datas = []
      for i in range(len(grasp_poses)):
        data_transformed = self.dataset.transform(copy.deepcopy(data),grasp_poses[i])
        input_data = torch.from_numpy(data_transformed['input'])
        input_datas.append(input_data)

      input_datas = torch.stack(input_datas,dim=0)
      n_split = int(np.ceil(len(input_datas)/batch_size))
      ids = np.arange(len(input_datas))
      ids_split = np.array_split(ids,n_split)

      out = []
      for i in range(len(ids_split)):
        ids = ids_split[i]
        input_data = input_datas[ids].cuda().float()
        pred = self.model(input_data)[0]
        pred = pred.softmax(dim=1).data.cpu().numpy()
        for b in range(len(pred)):
          cur_pred = pred[b]
          pred_label = cur_pred.argmax()
          confidence = cur_pred[pred_label]
          out.append([pred_label,confidence,cur_pred])
    torch.cuda.empty_cache()

    return out



class NunocsPredicter:
  def __init__(self,class_name):
    self.class_name = class_name
    self.class_name_to_artifact_id = {
      'nut': 78,
      'hnm': 73,
      'screw': 76
    }
    if self.class_name=='nut':
      self.min_scale = [0.005,0.005,0.001]
      self.max_scale = [0.05,0.05,0.05]
    elif self.class_name=='hnm':
      self.min_scale = [0.005,0.005,0.005]
      self.max_scale = [0.15,0.05,0.05]
    elif self.class_name=='screw':
      self.min_scale = [0.005,0.005,0.005]
      self.max_scale = [0.15,0.05,0.05]

    artifact_id = self.class_name_to_artifact_id[class_name]
    code_dir = os.path.dirname(os.path.realpath(__file__))
    artifact_dir = f"{code_dir}/artifacts/artifacts-{artifact_id}"
    print('NunocsPredicter artifact_dir',artifact_dir)
    with open(f"{artifact_dir}/config_nunocs.yml",'r') as ff:
      self.cfg = yaml.safe_load(ff)
    if os.path.exists('{}/normalizer.pkl'.format(artifact_dir)):
      with open('{}/normalizer.pkl'.format(artifact_dir),'rb') as ff:
        tmp = pickle.load(ff)
      self.cfg['mean'] = tmp['mean']
      self.cfg['std'] = tmp['std']
    self.dataset = NunocsIsolatedDataset(self.cfg,phase='test')

    self.model = PointNetSeg(n_in=self.cfg['input_channel'],n_out=3*self.cfg['ce_loss_bins'])

    self.model = load_model(self.model,ckpt_dir='{}/best_val.pth.tar'.format(artifact_dir))
    self.model.cuda().eval()


  def predict(self,data):
    with torch.no_grad():
      data['cloud_nocs'] = np.zeros(data['cloud_xyz'].shape)
      data['cloud_rgb'] = np.zeros(data['cloud_xyz'].shape)
      data_transformed = self.dataset.transform(copy.deepcopy(data))
      self.data_transformed = data_transformed
      ori_cloud = data_transformed['cloud_xyz_original']
      input_data = torch.from_numpy(data_transformed['input']).cuda().float().unsqueeze(0)

      pred = self.model(input_data)[0].reshape(-1,3,self.cfg['ce_loss_bins'])
      bin_resolution = 1/self.cfg['ce_loss_bins']
      pred_coords = pred.argmax(dim=-1).float()*bin_resolution
      probs = pred.softmax(dim=-1)
      confidence_z = torch.gather(probs[:,2,:],dim=-1,index=pred[:,2,:].argmax(dim=-1).unsqueeze(-1)).data.cpu().numpy().reshape(-1)
      conf_color = array_to_heatmap_rgb(confidence_z)
      nocs_cloud = pred_coords.data.cpu().numpy()-0.5

      nocs_cloud_down = copy.deepcopy(nocs_cloud)
      ori_cloud_down = copy.deepcopy(ori_cloud)

      best_ratio = 0
      best_transform = None
      best_nocs_cloud = None
      best_symmetry_tf = None
      for symmetry_tf in [np.eye(4)]:
        tmp_nocs_cloud_down = (symmetry_tf@to_homo(nocs_cloud_down).T).T[:,:3]
        for thres in [0.003,0.005]:
          use_kdtree_for_eval = False
          kdtree_eval_resolution = 0.003
          transform, inliers = estimate9DTransform(source=tmp_nocs_cloud_down,target=ori_cloud_down,PassThreshold=thres,max_iter=10000,use_kdtree_for_eval=use_kdtree_for_eval,kdtree_eval_resolution=kdtree_eval_resolution,max_scale=self.max_scale,min_scale=self.min_scale,max_dimensions=np.array([1.2,1.2,1.2]))
          if transform is None:
            continue

          if np.linalg.det(transform[:3,:3])<0:
            continue
          scales = np.linalg.norm(transform[:3,:3],axis=0)
          print("thres",thres)
          print("estimated scales",scales)
          print("transform:\n",transform)
          transformed = (transform@to_homo(tmp_nocs_cloud_down).T).T[:,:3]
          err_thres = 0.003

          cloud_at_canonical = (np.linalg.inv(transform)@to_homo(ori_cloud_down).T).T[:,:3]
          dimensions = cloud_at_canonical.max(axis=0)-cloud_at_canonical.min(axis=0)
          print("estimated canonical dimensions",dimensions)

          errs = np.linalg.norm(transformed-ori_cloud_down, axis=1)
          ratio = np.sum(errs<=err_thres)/len(errs)
          inliers = np.where(errs<=err_thres)[0]

          print("inlier ratio",ratio)

          if ratio>best_ratio:
            best_ratio = ratio
            best_symmetry_tf = symmetry_tf
            best_transform = transform.copy()
            best_nocs_cloud = copy.deepcopy(tmp_nocs_cloud_down)

      if best_transform is None:
        return None,None

      print(f"nocs predictor best_ratio={best_ratio}, scales={np.linalg.norm(best_transform[:3,:3],axis=0)}")
      print("nocs pose\n",best_transform)
      self.best_ratio = best_ratio
      transform = best_transform
      self.nocs_pose = transform.copy()
      nocs_cloud = (best_symmetry_tf@to_homo(nocs_cloud).T).T[:,:3]

      return nocs_cloud, transform


class PointGroupPredictor:
  def __init__(self,class_name):
    self.class_name_to_artifact_id = {
      'nut': 40,
      'hnm': 68,
      'screw': 77,
    }
    self.class_name = class_name
    artifact_id = self.class_name_to_artifact_id[class_name]
    code_dir = os.path.dirname(os.path.realpath(__file__))
    artifact_dir = f"{code_dir}/artifacts/artifacts-{artifact_id}"
    print('PointGroupPredictor artifact_dir',artifact_dir)
    config_dir = f"{artifact_dir}/config_pointgroup.yaml"
    self.cfg_pg = config_pg.get_parser(config_dir=config_dir)
    with open(config_dir,'r') as ff:
      self.cfg = yaml.safe_load(ff)

    self.dataset = dataset_seg.Dataset(cfg=self.cfg,cfg_pg=self.cfg_pg,phase='test')

    self.model = PointGroup(self.cfg_pg)
    self.model = load_model(self.model,ckpt_dir='{}/best_val.pth.tar'.format(artifact_dir))
    self.model.cuda().eval()

    self.n_slice_per_side = 1


  def predict(self,data):
    with torch.no_grad():
      xmax = data['cloud_xyz'][:,0].max()
      xmin = data['cloud_xyz'][:,0].min()
      ymax = data['cloud_xyz'][:,1].max()
      ymin = data['cloud_xyz'][:,1].min()
      xlen = (xmax-xmin)/self.n_slice_per_side
      ylen = (ymax-ymin)/self.n_slice_per_side

      batch_offsets = [0]
      locs = []
      xyz_original_all = []
      feats = []
      colors = []

      for ix in range(self.n_slice_per_side):
        for iy in range(self.n_slice_per_side):
          xstart = xmin+ix*xlen
          ystart = ymin+iy*ylen
          keep_mask = (data['cloud_xyz'][:,0]>=xstart) & (data['cloud_xyz'][:,0]<=xstart+xlen) & (data['cloud_xyz'][:,1]>=ystart) & (data['cloud_xyz'][:,1]<=ystart+ylen)
          xyz_origin = data['cloud_xyz'][keep_mask]
          normals = data['cloud_normal'][keep_mask]
          color = data['cloud_rgb'][keep_mask]

          pcd = toOpen3dCloud(xyz_origin)
          pcd = pcd.voxel_down_sample(voxel_size=self.cfg['downsample_size'])
          pts = np.asarray(pcd.points).copy()
          kdtree = cKDTree(xyz_origin)
          dists,indices = kdtree.query(pts)
          xyz_origin = xyz_origin[indices]
          normals = normals[indices]
          color = color[indices]

          xyz = xyz_origin * self.dataset.scale
          xyz -= xyz.min(0)
          batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

          i = ix+iy*self.n_slice_per_side
          locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))

          xyz_original_all.append(torch.from_numpy(xyz_origin))
          feats.append(torch.from_numpy(normals))
          colors.append(torch.from_numpy(color))

      batchsize = len(batch_offsets)-1
      batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
      locs = torch.cat(locs, 0)
      xyz_original_all = torch.cat(xyz_original_all, 0).to(torch.float32)
      feats = torch.cat(feats, 0)
      colors = torch.cat(colors, 0)

      spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.dataset.full_scale[0], None)

      voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, len(batch_offsets)-1, self.dataset.mode)

      coords = locs.cuda()
      voxel_coords = voxel_locs.cuda()
      p2v_map = p2v_map.cuda()
      v2p_map = v2p_map.cuda()

      coords_float = xyz_original_all.cuda().float()
      feats = feats.cuda().float()

      batch_offsets = batch_offsets.cuda()


      if self.cfg_pg.use_coords:
          feats = torch.cat((feats, coords_float), 1)
      voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, self.cfg_pg.mode)

      input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, self.cfg_pg.batch_size)

      ret = self.model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch=self.model.prepare_epochs-1)
      offsets = ret['pt_offsets'].data.cpu().numpy()
      xyz_original_all = xyz_original_all.data.cpu().numpy()

      pcd = toOpen3dCloud(xyz_original_all)
      pcd = pcd.voxel_down_sample(voxel_size=0.002)
      xyz_down = np.asarray(pcd.points).copy()
      kdtree = cKDTree(xyz_original_all)
      dists,indices = kdtree.query(xyz_down)
      xyz_down = xyz_original_all[indices]
      xyz_shifted = xyz_down+offsets[indices]
      self.xyz_shifted = xyz_shifted

      if self.class_name=='hnm':
        eps = 0.003
        min_samples = 20
        bandwidth = 0.005
      elif self.class_name=='nut':
        eps = 0.003
        min_samples = 5
        bandwidth = 0.007
      elif self.class_name=='screw':
        eps = 0.003
        min_samples = 5
        bandwidth = 0.009
      else:
        raise NotImplemented

      labels = MeanShift(bandwidth=bandwidth,cluster_all=True,n_jobs=-1,seeds=None).fit_predict(xyz_shifted)

      kdtree = cKDTree(xyz_down)
      dists,indices = kdtree.query(data['cloud_xyz'])
      labels_all = labels[indices]

      return labels_all
