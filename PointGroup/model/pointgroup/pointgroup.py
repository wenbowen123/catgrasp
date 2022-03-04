'''
PointGroup
Written by Li Jiang
'''

import torch,os,sys
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../../../')
sys.path.append('../../')
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from Utils import *

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=True)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=True, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=True, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        input_c = cfg.input_channel
        m = cfg.m
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-5, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=True, indice_key='subm1')
        )

        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, 3, bias=True),
        )

        self.score_unet = UBlock([m, 2*m], norm_fn, block_reps=2, block=block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)



    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, max_scale, batch_offsets, mode):
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=max_scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        box = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - box - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - box + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def forward(self, input_tensor, input_map, coords, batch_idxs, batch_offsets, epoch):
        ret = {}

        output = self.input_conv(input_tensor)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        pt_offsets = self.offset(output_feats)

        ret['pt_offsets'] = pt_offsets

        if(epoch > self.prepare_epochs):
            object_idxs = torch.arange(0,len(coords)).view(-1)
            semantic_preds_cpu = torch.ones(len(coords)).int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords + pt_offsets, batch_idxs, batch_offsets, self.cfg.cluster_radius_shift, self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)

            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()

            idx, start_len = pointgroup_ops.ballquery_batch_p(coords, batch_idxs, batch_offsets, self.cluster_radius, self.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_fullscale, self.cfg.scale, batch_offsets, self.mode)

            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()]
            score_feats = pointgroup_ops.roipool(score_feats, proposals_offset.cuda())
            scores = self.score_linear(score_feats)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        return ret



def model_fn_decorator(test=False):
    from util.config import get_parser
    cfg = get_parser()

    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='mean').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()
        voxel_coords = batch['voxel_locs'].cuda()
        p2v_map = batch['p2v_map'].cuda()
        v2p_map = batch['v2p_map'].cuda()

        coords_float = batch['locs_float'].cuda()
        feats = batch['feats'].cuda()

        batch_offsets = batch['offsets'].cuda()

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        semantic_scores = ret['semantic_scores']
        pt_offsets = ret['pt_offsets']
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']

        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

        return preds


    def model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()
        voxel_coords = batch['voxel_locs'].cuda()
        p2v_map = batch['p2v_map'].cuda()
        v2p_map = batch['v2p_map'].cuda()

        coords_float = batch['locs_float'].cuda()
        feats = batch['feats'].cuda()
        instance_labels = batch['instance_labels'].cuda()

        instance_info = batch['instance_info'].cuda()
        instance_pointnum = batch['instance_pointnum'].cuda()

        batch_offsets = batch['offsets'].cuda()

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)

        pt_offsets = ret['pt_offsets']
        if(epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']

        loss_inp = {}
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        if(epoch > cfg.prepare_epochs):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        with torch.no_grad():
            preds = {}
            preds['pt_offsets'] = pt_offsets.detach()
            if(epoch > cfg.prepare_epochs):
                preds['score'] = scores.detach()
                preds['proposals'] = (proposals_idx.detach(), proposals_offset.detach())

            visual_dict = {}
            visual_dict['loss'] = loss.item()
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        pt_offsets, coords_float, instance_info, instance_labels = loss_inp['pt_offsets']
        gt_offsets = instance_info[:,:3] - coords_float

        offset_norm_loss = nn.MSELoss()(pt_offsets,gt_offsets)
        offset_dir_loss = torch.tensor(0).cuda().float()

        loss_out['offset_norm_loss'] = (offset_norm_loss.item(), len(gt_offsets))
        loss_out['offset_dir_loss'] = (offset_dir_loss.item(), len(gt_offsets))

        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum)
            gt_ious, gt_instance_idxs = ious.max(1)
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            loss_out['score_loss'] = (score_loss.item(), gt_ious.shape[0])

        loss = cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss
        if(epoch > cfg.prepare_epochs):
            loss += (cfg.loss_weight[3] * score_loss)

        return loss, loss_out, infos


    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
