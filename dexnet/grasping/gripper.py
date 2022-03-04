# -*- coding: utf-8 -*-
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
Class to encapsulate robot grippers
Author: Jeff
"""
import open3d as o3d
import json,trimesh,torch
import numpy as np
import os
import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../../'.format(code_dir))
sys.path.append('{}/../../meshpy'.format(code_dir))
from meshpy.sdf_file import SdfFile
from autolab_core import RigidTransform
from Utils import *


class RobotGripper(object):
    """ Robot gripper wrapper for collision checking and encapsulation of grasp parameters (e.g. width, finger radius, etc)
    Note: The gripper frame should be the frame used to command the physical robot

    Attributes
    ----------
    name : :obj:`str`
        name of gripper
    mesh : :obj:`Mesh3D`
        3D triangular mesh specifying the geometry of the gripper
    params : :obj:`dict`
        set of parameters for the gripper, at minimum (finger_radius and grasp_width)
    T_grasp_gripper : :obj:`RigidTransform`
        transform from gripper frame to the grasp canonical frame (y-axis = grasp axis, x-axis = palm axis)
    """

    def __init__(self, gripper_folder, mesh_filename, params, T_grasp_gripper, sdf=None, sdf_enclosed=None):
        self.gripper_folder = gripper_folder
        self.trimesh = trimesh.load(mesh_filename)
        self.trimesh_enclosed = trimesh.load(mesh_filename.replace('_air_tight.obj','_enclosed_air_tight.obj'))
        self.mesh_filename = mesh_filename
        self.T_grasp_gripper = T_grasp_gripper
        self.sdf = sdf
        self.sdf_enclosed = sdf_enclosed

        self.finger_mesh1 = trimesh.load(f'{os.path.dirname(mesh_filename)}/finger1.obj')
        self.finger_mesh1_in_grasp = self.finger_mesh1.apply_transform(np.linalg.inv(self.get_grasp_pose_in_gripper_base()))
        self.finger_xmin = self.finger_mesh1_in_grasp.vertices[:,0].min()
        self.finger_xmax = self.finger_mesh1_in_grasp.vertices[:,0].max()
        self.finger_zmin = self.finger_mesh1_in_grasp.vertices[:,2].min()
        self.finger_zmax = self.finger_mesh1_in_grasp.vertices[:,2].max()
        self.finger_ymin = self.finger_mesh1_in_grasp.vertices[:,1].max()  #!NOTE y-axis is the finger close direction
        self.finger_ymax = -self.finger_ymin

        for key, value in list(params.items()):
            print("Gripper {}: {}".format(key,value))
            setattr(self, key, value)


    def get_grasp_pose_in_gripper_base(self):
        pose = np.eye(4)
        pose[:3,:3] = self.T_grasp_gripper.rotation
        pose[:3,3] = self.T_grasp_gripper.translation
        return np.linalg.inv(pose)


    def get_points_between_finger(self,pts_in_grasp):
        keep_mask = (pts_in_grasp[:,0]>=self.finger_xmin) & (pts_in_grasp[:,0]<=self.finger_xmax) & (pts_in_grasp[:,1]>=self.finger_ymin) & (pts_in_grasp[:,1]<=self.finger_ymax) & (pts_in_grasp[:,2]>=self.finger_zmin) & (pts_in_grasp[:,2]<=self.finger_zmax)
        return pts_in_grasp[keep_mask]


    @staticmethod
    def load(gripper_dir):
        """ Load the gripper specified by gripper_name.

        Parameters
        ----------
        gripper_dir : :obj:`str`
            relative path

        Returns
        -------
        :obj:`RobotGripper`
            loaded gripper objects
        """
        code_dir = os.path.dirname(os.path.realpath(__file__))
        gripper_dir = f"{code_dir}/../../{gripper_dir}"
        mesh_filename = os.path.join(gripper_dir, 'gripper_air_tight.obj')

        f = open(os.path.join(os.path.join(gripper_dir, 'params.json')), 'r')
        params = json.load(f)

        T_grasp_gripper = RigidTransform.load(os.path.join(gripper_dir, 'T_grasp_gripper.tf'))
        if T_grasp_gripper._from_frame=='gripper' and T_grasp_gripper._to_frame=='grasp':
            pass
        elif T_grasp_gripper._from_frame=='grasp' and T_grasp_gripper._to_frame=='gripper':
            T_grasp_gripper = T_grasp_gripper.inverse()
        else:
            raise RuntimeError("T_grasp_gripper from={}, to={}".format(T_grasp_gripper._from_frame,T_grasp_gripper._to_frame))
        new_gripper = RobotGripper(gripper_dir, mesh_filename, params, T_grasp_gripper)

        sdf_dir = mesh_filename.replace('.obj','.sdf')
        if os.path.exists(sdf_dir):
            sdf = SdfFile(sdf_dir).read()
            new_gripper.sdf = copy.deepcopy(sdf)

        sdf_dir = mesh_filename.replace('_air_tight.obj','_enclosed_air_tight.sdf')
        if os.path.exists(sdf_dir):
            print('sdf_dir',sdf_dir)
            sdf = SdfFile(sdf_dir).read()
            new_gripper.sdf_enclosed = copy.deepcopy(sdf)

        return new_gripper




def save_grasp_pose_mesh(gripper,grasp_pose,out_dir,enclosed=False):
  grasp_in_gripper = gripper.get_grasp_pose_in_gripper_base()
  if not enclosed:
    mesh = copy.deepcopy(gripper.trimesh)
  else:
    mesh = copy.deepcopy(gripper.trimesh_enclosed)
  mesh.apply_transform(grasp_pose@np.linalg.inv(grasp_in_gripper))
  mesh.export(out_dir)

