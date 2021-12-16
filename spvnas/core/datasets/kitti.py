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
        sample_stride = kwargs.get('sample_stride', 1)
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
                              split='train',),
            'valid':
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
                              split='valid'),
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
                              split='test')
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
                 sample_stride=1,):

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

        txt_path = '/'.join(self.data_path.split('/')[:-1])
        self.pcs = []
        self.crm_pcs = []
        self.planes = []
        self.labels = []
        self.calibs = []

        # weakly annotated data loader
        if split == 'train':
            train_idxs = open( os.path.join(root, txt_path, "train.txt") ).readlines()
            for idx in train_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join( self.root, self.data_path ,'%s.bin' % idx))
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
                self.planes.append(os.path.join(self.root, self.planes_path, '%s.txt' % idx) )
                self.labels.append( os.path.join(self.root, self.labels_path, '%s.txt' % idx) )
                self.calibs.append( os.path.join(self.root, self.calibs_path, '%s.txt' % idx) )
        elif split=="valid":
            val_idxs = open( os.path.join(root, txt_path, "val.txt") ).readlines()
            val_idxs = val_idxs[:len(val_idxs)//2]
            for idx in val_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join(self.root, self.data_path, '%s.bin' % idx))
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
                self.planes.append(os.path.join(self.root, self.planes_path, '%s.txt' % idx) )
                self.labels.append( os.path.join(self.root, self.labels_path, '%s.txt' % idx) )
                self.calibs.append( os.path.join(self.root, self.calibs_path, '%s.txt' % idx) )
        elif split=="test":
            #val_idxs = open( os.path.join(root, txt_path, "test500.txt") ).readlines()
            val_idxs = open( "/data/Ezgi/CenterPoint/data/kitti_prm/ImageSets_1000/train.txt" ).readlines()
            #val_idxs = open( "/data/Ezgi/testtt.txt" ).readlines()
            #val_idxs = val_idxs[len(val_idxs)//2:]
            for idx in val_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join(self.root, self.data_path, '%s.bin' % idx))
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
                self.planes.append(os.path.join(self.root, self.planes_path, '%s.txt' % idx) )
                self.labels.append( os.path.join(self.root, self.labels_path, '%s.txt' % idx) )
                self.calibs.append( os.path.join(self.root, self.calibs_path, '%s.txt' % idx) )

    def set_angle(self, angle):
        self.angle = angle

    def align_pcd(self, pc, plane_file):
        points = pc
        #get plane model from txt
        plane_model =  open(plane_file, 'r').readlines()[0].rstrip()
        a,b,c,d = map(float, plane_model.split(' '))
        plane_model = [a,b,c,d]

        pcd=open3d.open3d.geometry.PointCloud()
        pcd.points= open3d.open3d.utility.Vector3dVector(points[:, 0:3])

        # Translate plane to coordinate center
        pcd.translate((0,-d/c,0))

        # Calculate rotation angle between plane normal & z-axis
        plane_normal = tuple(plane_model[:3])
        z_axis = (0,0,1)
        rotation_angle = vector_angle(plane_normal, z_axis)

        # Calculate rotation axis
        plane_normal_length = math.sqrt(a**2 + b**2 + c**2)
        u1 = b / plane_normal_length
        u2 = -a / plane_normal_length
        rotation_axis = (u1, u2, 0)
        from pyquaternion import Quaternion
        q8c = Quaternion(axis=rotation_axis, angle=rotation_angle)
        R = q8c.rotation_matrix
        pcd.rotate(R, center=(0,0,0))
        
        #return 3rd row which is z, which is height
        return np.asarray(pcd.points)[:,2]
    
    
    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, index):
        pc_file = open ( self.pcs[index], 'rb')        
        block_ = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
        if   'test' not in self.split:
            front_idxs = block_[:,0]>0
            block_ = block_[front_idxs]

        if self.input_channels == 5:
             height = self.align_pcd(pc=block_, plane_file=self.planes[index])
             block_ = np.concatenate( (block_, height.reshape(-1,1)), axis=1)

        # Data augmentation
        scale_factor = 1.0
        rot_mat = np.array([[1,0, 0],
                            [0,1, 0],
                            [0, 0, 1]])

        if 'train' in self.split or 'valid' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05) # a float number
            rot_mat = np.array([[np.cos(theta),
                                 np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block_[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor

        crm_target_ = generate_CRM(radius=self.radius, points=block_ , labels_path=self.labels[index], calibs_path=self.calibs[index], rot_mat = rot_mat, scale_factor =scale_factor)
        
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
            'rot_mat': rot_mat,
            'scale_factor': scale_factor,
            'file_name': self.pcs[index].split('/')[-1] #e.g. 000000.bin 
        }


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))
