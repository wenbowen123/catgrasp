'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import torch,gzip,pickle
from torch.utils.data import DataLoader
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../../'.format(code_dir))
sys.path.append('{}/../'.format(code_dir))
from data_reader import DataReader
sys.path.append('../')

from util.config import get_parser
from lib.pointgroup_ops.functions import pointgroup_ops
from Utils import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self,cfg,phase,cfg_pg=None):
        super().__init__()
        self.phase = phase
        self.cfg = cfg
        self.reader = DataReader(cfg)

        if cfg_pg is None:
            cfg_pg = get_parser()

        self.full_scale = cfg_pg.full_scale
        self.scale = cfg_pg.scale
        self.max_npoint = cfg_pg.max_npoint
        self.mode = cfg_pg.mode
        code_dir = os.path.dirname(os.path.realpath(__file__))
        if phase=='train':
            self.files = sorted(glob.glob(f"{code_dir}/../../{self.cfg['train_root']}/*.pkl"))
        else:
            self.files = sorted(glob.glob(f"{code_dir}/../../{self.cfg['val_root']}/*.pkl"))

        print("phase: {}, num files={}".format(phase,len(self.files)))

    def __len__(self):
        return len(self.files)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label):
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)[0]

            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        return np.matmul(xyz, m)


    def crop(self, xyz):
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def __getitem__(self,index):
        return index


    def merge(self, id):
        locs = []
        locs_float = []
        feats = []
        instance_labels = []
        colors = []

        instance_infos = []
        instance_pointnum = []

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):
            with gzip.open(self.files[idx],'rb') as ff:
                data = pickle.load(ff)
            xyz_origin = data['cloud_xyz']
            normals = data['cloud_normal']
            instance_label = data['cloud_seg']
            color = data['cloud_rgb']

            pcd = toOpen3dCloud(xyz_origin)
            pcd = pcd.voxel_down_sample(voxel_size=self.cfg['downsample_size'])
            pts = np.asarray(pcd.points).copy()
            kdtree = cKDTree(xyz_origin)
            dists,indices = kdtree.query(pts)
            xyz_origin = xyz_origin[indices]
            normals = normals[indices]
            color = color[indices]
            instance_label = instance_label[indices]

            instance_label -= instance_label.min()
            new_instance_label = np.zeros(instance_label.shape)
            unique_labels = np.unique(instance_label)
            for i_label,label in enumerate(unique_labels):
                ids = np.where(instance_label==label)[0]
                new_instance_label[ids] = i_label
            instance_label = copy.deepcopy(new_instance_label).astype(int)

            xyz = xyz_origin * self.scale
            xyz -= xyz.min(0)

            inst_num, inst_infos = self.getInstanceInfo(xyz_origin, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]
            inst_pointnum = inst_infos["instance_pointnum"]


            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_origin))
            feats.append(torch.from_numpy(normals))
            colors.append(torch.from_numpy(color))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

        locs = torch.cat(locs, 0)
        locs_float = torch.cat(locs_float, 0).to(torch.float32)
        feats = torch.cat(feats, 0)
        colors = torch.cat(colors, 0)
        instance_labels = torch.cat(instance_labels, 0).long()

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, len(batch_offsets)-1, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'colors':colors}


