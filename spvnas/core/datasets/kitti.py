import json
import os
import os.path
import random
import sys
from collections import Sequence

import h5py
import numpy as np
import scipy
import scipy.interpolate
import scipy.ndimage
import torch
from numba import jit
import cv2
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

__all__ = ['KITTI']

class KITTI(dict):
    def __init__(self, root, data_path, crm_path, voxel_size,quantization_size, num_points, input_channels, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        if submit_to_server:
            super(KITTI, self).__init__({
                'train':
                    KITTIInternal(root,
                                  data_path,
                                  crm_path,
                                  voxel_size,
                                  quantization_size,
                                  num_points,
                                  input_channels,
                                  sample_stride=1,
                                  split='train',
                                  submit=True),
                'test':
                    KITTIInternal(root,
                                  data_path,
                                  crm_path,
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
                    KITTIInternal(root,
                                  data_path,
                                  crm_path,
                                  voxel_size,
                                  quantization_size,
                                  num_points,
                                  input_channels,
                                  sample_stride=1,
                                  split='train',
                                  google_mode=google_mode),
                'test':
                    KITTIInternal(root,
                                  data_path,
                                  crm_path,
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
                 root,
                 data_path,
                 crm_path,
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
        self.root = root
        self.data_path = data_path
        self.crm_path = crm_path
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
        if split == 'train':
            train_idxs = open( os.path.join(root, txt_path, "train.txt") ).readlines()
            for idx in train_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join( self.root, self.data_path ,'%s.bin' % idx))
                
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
        #elif split=="val":
        elif split=="test":
            val_idxs = open( os.path.join(root, txt_path, "val.txt") ).readlines()
            for idx in val_idxs:
                idx = idx.strip()
                self.pcs.append(os.path.join(self.root, self.data_path, '%s.bin' % idx))
                self.crm_pcs.append(os.path.join(self.root, self.crm_path, '%s.npy' % idx))
        """elif split=='test':
            files = os.listdir(os.path.join(root, self.data_path) )
            for name in files:
                self.pcs.append(self.root + self.data_path + "/" + name)
        """

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, index):
        
        pc_file = open ( self.pcs[index], 'rb')
        block_ = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
        
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
        
        labels_ = np.load( self.crm_pcs[index]).astype(float)
        if True:#'train' in self.split:
            # get the points only at front view
            # (x,y,z,r) -> (forward, left, up, r) since it's in Velodyne coords.
            front_idxs = block_[:,0]>=0
            block_ = block_[front_idxs] 
            labels_ = labels_[front_idxs]
                
        """
        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta),
                                 np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
            #block[:, 3:] = block_[:, 3:] + np.random.randn(3) * 0.1
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)
        """
        pc_ = np.round(block_[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = block_

        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True,
                                                    quantization_size=self.quantization_size)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds] #unique coords, # pc[inverse_map] = _pc
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc) #unique
        labels = SparseTensor(labels, pc) #unique
        labels_ = SparseTensor(labels_, pc_) #voxelized original
        inverse_map = SparseTensor(inverse_map, pc_) #voxelized orig
        
        #print("\nname: ", self.pcs[index].split('/')[-1])
        #print("size unique, original: ", pc.shape, pc_.shape)
        
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.pcs[index].split('/')[-1] #e.g. 000000.bin 
        }


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
