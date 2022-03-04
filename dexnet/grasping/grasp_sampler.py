# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import copy,os,sys,trimesh
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../../'.format(code_dir))
from dexnet.grasping.gripper import save_grasp_pose_mesh
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time
from scipy.spatial import cKDTree
import scipy.stats as stats
import dexnet

from dexnet.grasping.grasp import Grasp, ParallelJawPtGrasp3D
from dexnet.grasping.contacts import Contact3D
from autolab_core import RigidTransform
import scipy
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USE_OPENRAVE = True
try:
    import openravepy as rave
except ImportError:
    USE_OPENRAVE = False

try:
    import rospy
    import moveit_commander
    ROS_ENABLED = True
except ImportError:
    ROS_ENABLED = False

from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
try:
  multiprocessing.set_start_method('spawn')
except:
  pass
import my_cpp
from Utils import *


"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for sampling grasps.
Author: Jeff Mahler
"""


class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.

    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        self.friction_coef = config['sampling_friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        if 'max_num_surface_points' in list(config.keys()):
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 100
        if 'grasp_dist_thresh' in list(config.keys()):
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0


    def show_points(self, point, color='lb', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = color
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = color
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc=None, scale_factor=0.001,opacity=1.0):

        un2 = grasp_bottom_center
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
                      scale_factor=.05, line_width=0.25, color=(0, 1, 0), mode='arrow',opacity=opacity)
        if minor_pc is not None:
            mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
                      scale_factor=.05, line_width=0.1, color=(0, 0, 1), mode='arrow',opacity=opacity)
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
                      scale_factor=.05, line_width=0.05, color=(1, 0, 0), mode='arrow',opacity=opacity)



class PointConeGraspSampler(GraspSampler):
    def sample_grasps(self, background_pts,points_for_sample, normals_for_sample, max_num_samples=200, n_sphere_dir=100, approach_step=0.003, ee_in_grasp=None, cam_in_world=None, upper=None,lower=None,open_gripper_collision_pts=None, center_ob_between_gripper=False,filter_ik=True,adjust_collision_pose=True,**kwargs):
        resolution = compute_cloud_resolution(points_for_sample)
        print(f"estimated resolution={resolution}")
        self.params = {
            'debug_vis': False,
            'r_ball': resolution*3,
        }
        self.approach_step = approach_step

        sphere_pts = hinter_sampling(min_n_pts=1000,radius=1)[0]
        sphere_pts = sphere_pts/np.linalg.norm(sphere_pts,axis=-1).reshape(-1,1)
        higher_mask = sphere_pts[:,2]>=np.cos(60*np.pi/180)
        sphere_pts = sphere_pts[higher_mask]
        rot_y_180 = euler_matrix(0,np.pi/2,0,axes='sxyz')[:3,:3]
        sphere_pts = (rot_y_180@sphere_pts.T).T
        if sphere_pts.shape[0]>n_sphere_dir:
            ids = np.random.choice(np.arange(len(sphere_pts)),size=n_sphere_dir,replace=False)
            sphere_pts = sphere_pts[ids]
        print('#sphere_pts={}'.format(len(sphere_pts)))

        sample_ids = np.arange(len(points_for_sample))
        np.random.shuffle(sample_ids)
        if len(sample_ids)>max_num_samples:
            sample_ids = sample_ids[:max_num_samples]
        print("#sample_ids={}".format(len(sample_ids)))

        grasps = []
        seed = np.random.get_state()[1][0]

        sample_ids_splits = [sample_ids]
        n_sampled = 0

        grasps = []
        for id in sample_ids:
            tmp_grasps = self.sample_one_surface_point(points_for_sample[id],normals_for_sample[id],points_for_sample,normals_for_sample,background_pts,sphere_pts,seed)
            grasps += tmp_grasps


        if center_ob_between_gripper:
            print("begin center_ob_between_gripper...")
            for i in range(len(grasps)):
                g = grasps[i]
                pts_in_grasp = (np.linalg.inv(g.grasp_pose)@to_homo(points_for_sample).T).T[:,:3]
                pts_center_in_grasp = (pts_in_grasp.max(axis=0)+pts_in_grasp.min(axis=0)) / 2
                grasp_offset = np.eye(4)
                grasp_offset[:3,3] = [0,pts_center_in_grasp[1],0]
                grasps[i].grasp_pose = grasps[i].grasp_pose@grasp_offset


        grasp_poses = []
        for g in grasps:
            grasp_poses.append(g.grasp_pose)
        symmetry_tfs = [np.eye(4)]
        nocs_pose = np.eye(4)
        canonical_to_nocs = np.eye(4)
        gripper_in_grasp = np.linalg.inv(self.gripper.get_grasp_pose_in_gripper_base())
        filter_approach_dir_face_camera = True
        resolution = 0.0005
        verbose = True
        print(f"Filtering #grasp_poses={len(grasp_poses)}")
        grasp_poses = my_cpp.filterGraspPose(grasp_poses,list(symmetry_tfs),nocs_pose,canonical_to_nocs,cam_in_world,ee_in_grasp,gripper_in_grasp,filter_approach_dir_face_camera,filter_ik,adjust_collision_pose,upper,lower,self.gripper.trimesh.vertices,self.gripper.trimesh.faces,self.gripper.trimesh_enclosed.vertices,self.gripper.trimesh_enclosed.faces,open_gripper_collision_pts,background_pts,resolution,verbose)
        grasps = []
        for i in range(len(grasp_poses)):
            grasp = ParallelJawPtGrasp3D(grasp_pose=grasp_poses[i])
            grasps.append(grasp)

        return grasps


    def sample_one_surface_point(self,selected_surface,selected_normal,points_for_sample,normals_for_sample,background_pts,sphere_pts,seed=None):
        np.random.seed(seed)
        r_ball = self.params['r_ball']

        M = np.zeros((3, 3))
        point_cloud_kdtree = cKDTree(points_for_sample)
        kd_indices = point_cloud_kdtree.query_ball_point(selected_surface.reshape(1,3),r=r_ball)
        kd_indices = np.array(kd_indices[0]).astype(int).reshape(-1)
        sqr_distances = np.linalg.norm(selected_surface.reshape(1,3)-points_for_sample[kd_indices], axis=-1) ** 2

        for _ in range(len(kd_indices)):
            if sqr_distances[_] != 0:
                normal = normals_for_sample[kd_indices[_]]
                normal = normal.reshape(-1, 1)
                if np.linalg.norm(normal) != 0:
                    normal /= np.linalg.norm(normal)
                M += np.matmul(normal, normal.T)
        if sum(sum(M)) == 0:
            print("M matrix is empty as there is no point near the neighbour")
            self.params['r_ball'] *= 2
            print(f"Here is a bug, if points amount is too little it will keep trying and never go outside. Update r_ball to {self.params['r_ball']} and resample")
            return self.sample_one_surface_point(selected_surface,selected_normal,points_for_sample,normals_for_sample,background_pts,sphere_pts,seed)

        approach_normal = -selected_normal.reshape(3)
        approach_normal /= np.linalg.norm(approach_normal)
        eigval, eigvec = np.linalg.eig(M)
        def proj(u,v):
            u = u.reshape(-1)
            v = v.reshape(-1)
            return np.dot(u,v)/np.dot(u,u) * u
        minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)
        minor_pc = minor_pc-proj(approach_normal,minor_pc)
        minor_pc /= np.linalg.norm(minor_pc)
        major_pc = np.cross(minor_pc, approach_normal)
        major_pc = major_pc / np.linalg.norm(major_pc)

        if self.params['debug_vis']:
            self.show_grasp_norm_oneside(selected_surface, grasp_normal=approach_normal, grasp_axis=major_pc, minor_pc=minor_pc, scale_factor=0.001)
            self.show_points(selected_surface, color='g', scale_factor=0.005)
            self.show_points(points_for_sample,scale_factor=0.002)
            selected_id = np.linalg.norm(points_for_sample-selected_surface.reshape(1,3),axis=1).argmin()
            self.show_line(selected_surface, (selected_surface + normals_for_sample[selected_id]*0.05).reshape(3))
            mlab.show()

        R0 = np.concatenate((approach_normal.reshape(3,1), major_pc.reshape(3,1), minor_pc.reshape(3,1)), axis=1)
        Rs = [R0]
        for sphere_pt in sphere_pts:
            tmp_dir = sphere_pt.copy()
            R_sphere = directionVecToRotation(direction=tmp_dir,ref=np.array([1,0,0]))
            for x_rot in np.arange(0,180,30):
                R_inplane = euler_matrix(x_rot*np.pi/180,0,0,axes='sxyz')[:3,:3]
                Rs.append(R0@R_sphere@R_inplane)

        grasp_poses = []
        for R in Rs:
            R = normalizeRotation(R)
            if R is None or np.iscomplex(R).any():
                continue
            approach_dir = R[:,0]
            for d in np.arange(0,self.gripper.hand_depth,self.approach_step):
                grasp_center = selected_surface+self.gripper.init_bite*approach_dir+approach_dir*d
                grasp_pose = np.eye(4)
                grasp_pose[:3,:3] = R
                grasp_pose[:3,3] = grasp_center
                grasp_poses.append(grasp_pose)
        grasp_poses = np.array(grasp_poses)

        grasps = []
        for grasp_pose in grasp_poses:
            grasp = ParallelJawPtGrasp3D()
            grasp.grasp_pose = grasp_pose.copy()
            grasps.append(grasp)

        return grasps



class NocsTransferGraspSampler(GraspSampler):
    def __init__(self, gripper, config, canonical, class_name, score_larger_than=0, max_n_grasp=None, center_ob_between_gripper=False):
        super().__init__(gripper,config)
        self.canonical = copy.deepcopy(canonical)
        n_before = len(self.canonical['canonical_grasps'])
        self.class_name = class_name
        new_grasps = []
        if score_larger_than>0:
            for grasp in self.canonical['canonical_grasps']:
                if grasp.perturbation_score>=score_larger_than:
                    new_grasps.append(grasp)
            self.canonical['canonical_grasps'] = new_grasps
        if max_n_grasp is not None:
            self.canonical['canonical_grasps'].sort(key=lambda x:-x.perturbation_score)
            self.canonical['canonical_grasps'] = self.canonical['canonical_grasps'][:max_n_grasp]

        if center_ob_between_gripper:
            print('center_ob_between_gripper...')
            for i in range(len(self.canonical['canonical_grasps'])):
                grasp_in_ob = self.canonical['canonical_grasps'][i].get_grasp_pose_matrix()
                ob_in_grasp = np.linalg.inv(grasp_in_ob)
                ob_in_grasp[1,3] = 0
                new_grasp_pose = np.linalg.inv(ob_in_grasp)
                self.canonical['canonical_grasps'][i].grasp_pose = new_grasp_pose

        print(f"NocsTransferGraspSampler score_larger_than={score_larger_than}, center_ob_between_gripper={center_ob_between_gripper}, max_n_grasp={max_n_grasp}, #canonical_grasp={len(self.canonical['canonical_grasps'])}, before has {n_before}")


    def sample_grasps(self, background_pts,open_gripper_collision_pts, normals_for_sample, nocs_pts, nocs_pose, cam_in_world,ee_in_grasp,upper,lower,filter_approach_dir_face_camera=False,ik_func=None,symmetry_tfs=[np.eye(4)],filter_ik=True,**kwargs):
        canonical_to_nocs = np.eye(4)

        resolution = 0.0005
        grasp_pose_in_gripper = self.gripper.get_grasp_pose_in_gripper_base()

        grasps = []

        grasp_poses = []
        for igrasp,grasp in enumerate(self.canonical['canonical_grasps']):
            grasp_poses.append(grasp.get_grasp_pose_matrix())
        gripper_in_grasp = np.linalg.inv(self.gripper.get_grasp_pose_in_gripper_base())
        print('grasp_poses before filter',len(grasp_poses)*len(symmetry_tfs))
        verbose = False
        adjust_collision_pose = True
        grasp_poses = my_cpp.filterGraspPose(grasp_poses,list(symmetry_tfs),nocs_pose,canonical_to_nocs,cam_in_world,ee_in_grasp,gripper_in_grasp,filter_approach_dir_face_camera,filter_ik,adjust_collision_pose,upper,lower,self.gripper.trimesh.vertices,self.gripper.trimesh.faces,self.gripper.trimesh_enclosed.vertices,self.gripper.trimesh_enclosed.faces,open_gripper_collision_pts,background_pts,resolution,verbose)

        print('#grasp_poses with symmetry and after filter',len(grasp_poses))

        grasp_poses = np.array(grasp_poses)
        for i in range(len(grasp_poses)):
            grasp = ParallelJawPtGrasp3D(grasp_pose=grasp_poses[i])
            grasps.append(grasp)

        print("Sampled grasps: {}".format(len(grasps)))

        return grasps



class CombinedGraspSampler(GraspSampler):
    def __init__(self,gripper,config,samplers=[]):
        super().__init__(gripper,config)
        self.samplers = samplers

    def sample_grasps(self,**kwargs):
        grasps_all = []
        for sampler in self.samplers:
            grasps = sampler.sample_grasps(**kwargs)
            grasps_all += grasps
        return grasps_all