import os
import numpy as np
from .config import *

from .bev_generator import *
from .calibration import *
from torch.utils.data import Dataset, DataLoader
import torch

class KITTI_Dataset(Dataset):

    def __init__(self, mode='training',  use_val=True, num_points=100000):

        self.use_val = use_val
        self.mode = mode
        self.num_points = num_points
        self.pc_paths = []
        self.crm_pc_paths = []
        self.crm_label_paths = []
        self.calib_paths = []
        if mode=='training':
            if not use_val:
                self.pc_paths = os.listdir(os.path.join(root_dir, data_train_path) )
                self.crm_pc_paths = os.listdir(os.path.join(root_dir, crm_train_path_pc) )
                self.labels_pc_path = os.listdir(os.path.join(root_dir, crm_train_path_labels) )
                self.calib_paths.append(os.path.join(root_dir, calib_train_path))

            else: # only data in train.txt will be used as training data
                train_idxs = open( os.path.join(root_dir, "train.txt") ).readlines()
                for idx in train_idxs:
                    idx = idx.strip()
                    self.pc_paths.append('%s.bin' % idx)
                    self.crm_pc_paths.append('%s.npy' % idx)
                    self.crm_label_paths.append('%s.txt' % idx)
                    self.calib_paths.append('%s.txt' % idx)

        elif mode=='testing':
            if use_val:
                val_idxs = open( os.path.join(root_dir, "val.txt") ).readlines()
                for idx in val_idxs:
                    idx = idx.strip()
                    self.pc_paths.append('%s.bin' % idx)
                    self.crm_pc_paths.append('%s.npy' % idx)
                    self.crm_label_paths.append('%s.txt' % idx)
                    self.calib_paths.append('%s.txt' % idx)
            else:
                self.pc_paths = os.listdir(os.path.join(root_dir, data_test_path) )
                self.calib_paths.append(os.path.join(root_dir, calib_test_path))


    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        if self.mode=='testing' and not self.use_val:

            pc_file = open (os.path.join(root_dir, data_test_path, self.pc_paths[idx]), 'rb')
            pc = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]

            calibs = Calibration( os.path.join(root_dir, calib_train_path), self.calib_paths[idx])
            sample = {'pc': pc, 'crm_pc': None}# 'crm_labels': None}

        else:
            pc_file = open (os.path.join(root_dir, data_train_path, self.pc_paths[idx]), 'rb')
            pc = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]

            crm_pc = np.load(os.path.join(root_dir, crm_train_path_pc, self.crm_pc_paths[idx]))

            with open( os.path.join(root_dir, crm_train_path_labels, self.crm_label_paths[idx]) ) as f:
                crm_labels = f.readlines()
                crm_labels = [x.strip() for x in crm_labels]

            calibs = Calibration( os.path.join(root_dir, calib_train_path), self.calib_paths[idx])

            sample = {'pc': pc, 'crm_pc': crm_pc}# 'crm_labels': crm_labels}

        #pc, crm_pc, _ = bev_generator.generate_BEV_CRM_and_Z(pc, crm_pc, res=0.2)
        pc, crm_pc = bev_generator.generate_BEV_matrix(pc, crm_pc, num_points)
        #pc = np.expand_dims(pc, axis=0)
        #crm_pc = np.expand_dims(crm_pc, axis=0)
        pc = torch.from_numpy(pc).float()
        crm_pc = torch.from_numpy(crm_pc).float()

        sample['pc'] = pc
        sample['crm_pc'] = crm_pc
        

        return sample