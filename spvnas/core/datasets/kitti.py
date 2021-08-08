import json
import os
import os.path
import random
import sys
from collections import Sequence
import math
import h5py
import numpy as np
import scipy
import scipy.interpolate
import scipy.ndimage
import torch
from numba import jit
import cv2
import open3d
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from .utils import *

__all__ = ['KITTI']

class KITTI(dict):
    def __init__(self, radius, root, data_path, crm_path, labels_path, calibs_path, planes_path, voxel_size,quantization_size, num_points, input_channels, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        if submit_to_server:
            super(KITTI, self).__init__({
                'train':
                    KITTIInternal(radius,
                                  root,
                                  data_path,
                                  crm_path,
                                  labels_path,
                                  calibs_path,
                                  planes_path,
                                  voxel_size,
                                  quantization_size,
                                  num_points,
                                  input_channels,
                                  sample_stride=1,
                                  split='train',
                                  submit=True),
                'test':
                    KITTIInternal(radius,
                                  root,
                                  data_path,
                                  crm_path,
                                  labels_path,
                                  calibs_path,
                                  planes_path,
                                  voxel_size,
                                  quantization_size,
                                  num_points,
                                  input_channels,
                                  sample_stride=1,
                                  split='test')
            })
        else:
            super(KITTI, self).__init__({
                'train':
                    KITTIInternal(radius,
                                  root,
                                  data_path,
                                  crm_path,
                                  labels_path,
                                  calibs_path,
                                  planes_path,
                                  voxel_size,
                                  quantization_size,
                                  num_points,
                                  input_channels,
                                  sample_stride=1,
                                  split='train',
                                  google_mode=google_mode),
                'test':
                    KITTIInternal(radius,
                                  root,
                                  data_path,
                                  crm_path,
                                  labels_path,
                                  calibs_path,
                                  planes_path,
                                  voxel_size,
                                  quantization_size,
                                  num_points,
                                  input_channels,
                                  sample_stride=sample_stride,
                                  split='test')  #,
                #'real_test': KITTIInternal(root, voxel_size, num_points, split='test')
            })


class KITTIInternal:
    def __init__(self,
                 radius,
                 root,
                 data_path,
                 crm_path,
                 labels_path,
                 calibs_path,
                 planes_path,
                 voxel_size,
                 quantization_size,
                 num_points,
                 input_channels,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
        if submit:
            trainval = True
        else:
            trainval = False
        self.radius = radius
        self.root = root
        self.data_path = data_path
        self.crm_path = crm_path
        self.labels_path = labels_path
        self.calibs_path = calibs_path
        self.planes_path = planes_path
        self.split = split
        self.voxel_size = voxel_size
        self.quantization_size = quantization_size
        self.num_points = num_points
        self.input_channels = input_channels
        self.sample_stride = sample_stride
        self.google_mode = google_mode

        txt_path = '/'.join(self.data_path.split('/')[:-1])
        self.pcs = []
        self.crm_pcs = []
        self.planes = []
        self.labels = []
        self.calibs = []
        if split == 'train':
            train_idxs = open( os.path.join(root, txt_path, "train.txt") ).readlines()
            for idx in train_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join( self.root, self.data_path ,'%s.bin' % idx))
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
                self.planes.append(os.path.join(self.root, self.planes_path, '%s.txt' % idx) )
                self.labels.append( os.path.join(self.root, self.labels_path, '%s.txt' % idx) )
                self.calibs.append( os.path.join(self.root, self.calibs_path, '%s.txt' % idx) )
        #elif split=="val":
        elif split=="test":
            val_idxs = open( os.path.join(root, txt_path, "val.txt") ).readlines()
            for idx in val_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join(self.root, self.data_path, '%s.bin' % idx))
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
                self.planes.append(os.path.join(self.root, self.planes_path, '%s.txt' % idx) )
                self.labels.append( os.path.join(self.root, self.labels_path, '%s.txt' % idx) )
                self.calibs.append( os.path.join(self.root, self.calibs_path, '%s.txt' % idx) )
        """elif split=='test':
            files = os.listdir(os.path.join(root, self.data_path) )
            for name in files:
                self.pcs.append(self.root + self.data_path + "/" + name)
        """

    def set_angle(self, angle):
        self.angle = angle

    def align_pcd(self, pc_file, plane_file):
        pc_file = open ( pc_file, 'rb')
        points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

        #get plane model from txt
        plane_model =  open(plane_file, 'r').readlines()[3].rstrip()
        a,b,c,d = map(float, plane_model.split(' '))
        plane_model = [a,b,c,d]

        pcd=open3d.open3d.geometry.PointCloud()
        pcd.points= open3d.open3d.utility.Vector3dVector(points[:, 0:3])
        # Translate plane to coordinate center
        #pcd.translate((0,-d/c,0))

        # Calculate rotation angle between plane normal & z-axis
        plane_normal = tuple(plane_model[:3])
        z_axis = (0,0,1)
        rotation_angle = vector_angle(plane_normal, z_axis)

        # Calculate rotation axis
        plane_normal_length = math.sqrt(a**2 + b**2 + c**2)
        u1 = b / plane_normal_length
        u2 = -a / plane_normal_length
        rotation_axis = (u1, u2, 0)

        # Generate axis-angle representation
        optimization_factor = 1#1.4
        axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])

        # Rotate point cloud
        R = pcd.get_rotation_matrix_from_axis_angle(axis_angle)
        pcd.rotate(R, center=(0,0,0))

        points[:, :3] = np.asarray(pcd.points)
        return points

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, index):

        block_ = self.align_pcd(pc_file= self.pcs[index], plane_file=self.planes[index])
        calibs = read_calibs( self.calibs[index])
        instance_labels = read_labels(self.labels[index])
        #crm_target_ = np.load( self.crm_pcs[index]).astype(float)

        # Data augmentation
        scale_factor = 1.0
        rot_mat = np.array([[1,0, 0],
                            [0,1, 0],
                            [0, 0, 1]])

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05) # a float number
            rot_mat = np.array([[np.cos(theta),
                                 np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block_[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor

        crm_target_ = generate_CRM_wfiles(radius=self.radius, points=block_ , labels_path=self.labels[index], calibs_path=self.calibs[index],
                                   rot_mat = rot_mat, scale_factor =scale_factor)

        if True:# 'train' in self.split:
            # get the points only at front view
            # (x,y,z,r) -> (forward, left, up, r) since it's in Velodyne coords.
            front_idxs = block_[:,0]>=0
            block_ = block_[front_idxs] 
            crm_target_ = crm_target_[front_idxs]
                
        """ 
        if self.input_channels == 5:
            pcd=open3d.open3d.geometry.PointCloud()
            pcd.points= open3d.open3d.utility.Vector3dVector(block_[:, 0:3])
            #inlers contains the indexes of gorund points
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.15,
                                                     ransac_n=100,
                                                     num_iterations=1000)
            ground_feature = np.ones(shape=(block_.shape[0],1))
            ground_feature[inliers] = 0.0
            
            block_ = np.concatenate( (block_, ground_feature), axis=1)
        """
        pc_ = np.round(block_[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = block_

        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    crm_target_,
                                                    return_index=True,
                                                    return_invs=True,
                                                    quantization_size=self.quantization_size)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds] #unique coords, # pc[inverse_map] = _pc
        feat = feat_[inds]
        crm_target = crm_target_[inds]
        lidar = SparseTensor(feat, pc) #unique
        crm_target = SparseTensor(crm_target, pc) #unique
        crm_target_ = SparseTensor(crm_target_, pc_) #voxelized original
        inverse_map = SparseTensor(inverse_map, pc_) #voxelized orig
        
        return {
            'lidar': lidar,
            'targets': crm_target,
            'targets_mapped': crm_target_,
            'inverse_map': inverse_map,
            'calibs': self.calibs[index],
            'labels': self.labels[index],
            'pc_file': self.pcs[index],
            'rot_mat': rot_mat,
            'scale_factor': scale_factor,
            'subsize': len(inds),
            'file_name': self.pcs[index].split('/')[-1] #e.g. 000000.bin 
        }


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))
