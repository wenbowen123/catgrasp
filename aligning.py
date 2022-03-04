'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
RANSAC for Similarity Transformation Estimation

Written by Srinath Sridhar. Modified by Bowen.
'''
import open3d as o3d
import numpy as np
import cv2,yaml
import itertools
from Utils import *
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools



def estimateAffine3D(source,target,PassThreshold):
    '''
    @source: (N,3)
    '''
    ret,transform,inliers = cv2.estimateAffine3D(source, target,confidence=0.999,ransacThreshold=PassThreshold)
    tmp = np.eye(4)
    tmp[:3] = transform
    transform = tmp
    inliers = np.where(inliers>0)[0]
    return transform, inliers


def estimate9DTransform_worker(cur_src,cur_dst,source,target,PassThreshold,use_kdtree_for_eval=False,kdtree_eval_resolution=None,max_scale=np.array([99,99,99]),min_scale=np.array([0,0,0]),max_dimensions=None):
    bad_return = None,None,None
    transform,inliers = estimateAffine3D(source=cur_src,target=cur_dst,PassThreshold=PassThreshold)
    new_transform = transform.copy()
    scales = np.linalg.norm(transform[:3,:3],axis=0)
    if (scales>max_scale).any() or (scales<min_scale).any():
        return bad_return

    R = transform[:3,:3]/scales.reshape(1,3)
    u,s,vh = np.linalg.svd(R)

    if s.min()<0.8 or s.max()>1.2:
        return bad_return

    R = u@vh
    if np.linalg.det(R)<0:
        return bad_return

    new_transform[:3,:3] = R@np.diag(scales)
    transform = new_transform.copy()

    if max_dimensions is not None:
        cloud_at_canonical = (np.linalg.inv(transform)@to_homo(target).T).T[:,:3]
        dimensions = cloud_at_canonical.max(axis=0)-cloud_at_canonical.min(axis=0)
        if (dimensions>max_dimensions).any():
            return bad_return

    src_transformed = (transform@to_homo(source).T).T[:,:3]

    if not use_kdtree_for_eval:
        errs = np.linalg.norm(src_transformed-target,axis=-1)
        ratio = np.sum(errs<=PassThreshold)/len(errs)
        inliers = np.where(errs<=PassThreshold)[0]
    else:
        pcd = toOpen3dCloud(target)
        pcd = pcd.voxel_down_sample(voxel_size=kdtree_eval_resolution)
        kdtree = cKDTree(np.asarray(pcd.points).copy())
        dists1,indices1 = kdtree.query(src_transformed)
        pcd = toOpen3dCloud(src_transformed)
        pcd = pcd.voxel_down_sample(voxel_size=kdtree_eval_resolution)
        kdtree = cKDTree(np.asarray(pcd.points).copy())
        dists2,indices2 = kdtree.query(target)
        errs = np.concatenate((dists1,dists2),axis=0).reshape(-1)
        ratio = np.sum(errs<=PassThreshold)/len(errs)
        inliers = np.where(dists1<=PassThreshold)[0]

    return ratio,transform,inliers

def estimate9DTransform(source,target,PassThreshold,max_iter=1000,use_kdtree_for_eval=False,kdtree_eval_resolution=None,max_scale=np.array([99,99,99]),min_scale=np.array([0,0,0]),max_dimensions=None):
    best_transform = None
    best_ratio = 0
    inliers = None

    n_iter = 0
    srcs = []
    dsts = []
    for i in range(max_iter):
        ids = np.random.choice(len(source),size=4,replace=False)
        cur_src = source[ids]
        cur_dst = target[ids]
        srcs.append(cur_src)
        dsts.append(cur_dst)

    outs = []
    for i in range(len(srcs)):
        out = estimate9DTransform_worker(srcs[i],dsts[i],source,target,PassThreshold,use_kdtree_for_eval,kdtree_eval_resolution=kdtree_eval_resolution,max_scale=max_scale,min_scale=min_scale,max_dimensions=max_dimensions)
        if out[0] is None:
            continue
        outs.append((out))
    if len(outs)==0:
        return None,None

    ratios = []
    transforms = []
    inlierss = []
    for out in outs:
        ratio,transform,inliers = out
        ratios.append(ratio)
        transforms.append(transform)
        inlierss.append(inliers)

    best_id = np.array(ratios).argmax()
    best_transform = transforms[best_id]
    inliers = inlierss[best_id]
    return best_transform,inliers


def getRANSACInliers(source, target, n_sample=3, MaxIterations=100, PassThreshold=None, est_scale=True):
    '''
    @source: (N,3)
    '''
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))  #(4,N)
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    BestInlierRatio = 0
    BestInlierIdx = np.arange(3)
    best_res_vec = None
    for i in range(0, MaxIterations):
        RandIdx = np.random.choice(np.arange(SourceHom.shape[1]), size=n_sample,replace=False)
        Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx], est_scale=est_scale)
        if not np.isfinite(OutTransform).all():
            continue
        ResidualVec, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        if InlierRatio>BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
            best_res_vec = ResidualVec
    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio, BestInlierIdx


def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    InlierIdx = np.where(ResidualVec < PassThreshold)[0]
    if len(InlierIdx)<5:
        return ResidualVec,0,np.arange(3)

    nInliers = np.sum(ResidualVec < PassThreshold)
    InlierRatio = nInliers / float(SourceHom.shape[1])
    return ResidualVec, InlierRatio, InlierIdx


def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual


def estimateSimilarityUmeyama(SourceHom, TargetHom, est_scale=True):
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()  #(3,N)
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        raise RuntimeError('There are NANs in the input. nPoints={}'.format(nPoints))

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T # Transpose is the one that works

    ScaleFact = np.eye(3)
    if est_scale:
        varP = np.var(SourceHom[:3, :], axis=1).sum()
        ScaleFact = 1/varP * np.sum(D) # scale factor
        ScaleFact = np.diag([ScaleFact,ScaleFact,ScaleFact])

    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact@Rotation)

    Rotation = Rotation.T
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleFact@Rotation
    OutTransform[:3, 3] = Translation
    return ScaleFact, Rotation, Translation, OutTransform

