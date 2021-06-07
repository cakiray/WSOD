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
    def __init__(self, root, data_path, crm_path, voxel_size,quantization_size, num_points, **kwargs):
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
        self.sample_stride = sample_stride
        self.google_mode = google_mode

        self.pcs = []
        self.crm_pcs = []
        if split == 'train':
            train_idxs = open( os.path.join(root, "train.txt") ).readlines()
            for idx in train_idxs:
                idx = idx.strip()
                self.pcs.append( self.root + self.data_path + '/%s.bin' % idx)
                self.crm_pcs.append(self.root + self.crm_path + '/%s.npy' % idx)
        #elif split=="val":
        elif split=="test":
            val_idxs = open( os.path.join(root, "val.txt") ).readlines()
            #import random
            #test_idxs = random.sample(range(0, len(val_idxs)), 50)
            #for idx in test_idxs:
                #idx = val_idxs[idx].strip()
            val_idxs.sort()
            for idx in val_idxs[:50]:
                idx = idx.strip()
                self.pcs.append(self.root + self.data_path + '/%s.bin' % idx)
                self.crm_pcs.append(self.root + self.crm_path  + '/%s.npy' % idx)
        """elif split=='test':
            files = os.listdir(os.path.join(root, self.data_path) )
            for name in files:
                self.pcs.append(self.root + self.data_path + "/" + name)
        """
        """
        self.seqs = []

            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
            if self.google_mode or trainval:
                self.seqs.append('08')
            #if trainval is True:
            #    self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21'
            ]


    for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        #self.files = self.files[:40]
        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0
        # print('There are %d classes.' % (cnt))
        """
    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, index):
        
        pc_file = open ( self.pcs[index], 'rb')
        block_ = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
        labels_ = np.load( self.crm_pcs[index]).astype(float)
    
        # get the points only at front view
        # (x,y,z,r) -> (forward, left, up, r) since it's in Velodyne coords.
        front_idxs = block_[:,0]>=0
        block_ = block_[front_idxs] 
        labels_ = labels_[front_idxs]
            
        #print("\n name ", self.pcs[index].split('/')[-1])
        #print("\n original size ", block_.shape)
        #with open(self.pcs[index], 'rb') as b:
        #    block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        #print("block: ", block_)
        #block = np.zeros_like(block_)
        #block = block_
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
        #block[:, 3] = block_[:, 3]
        
        pc_ = np.round(block_[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)
        #inds = self.inds[index]

        """
        label_file = self.files[index].replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros((pc_.shape[0])).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
        inst_labels_ = (all_labels >> 16).astype(np.int64)  # instance labels
        """
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
            'file_name': self.pcs[index].split('/')[-1]
        }


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)