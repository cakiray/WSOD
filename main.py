# This is a sample Python script.
from centerresponsegeneration.utils import visualize_pointcloud, read_car_labels
import os
import torch
import numpy as np
from centerresponsegeneration.config import *
from centerresponsegeneration.calibration import Calibration
from spvnas.core.utils import *
from centerresponsegeneration.utils import visualize_pointcloud_onlypreds
if __name__ == '__main__':

    crm = np.load( os.path.join(root_dir, crm_train_path_pc, file)).astype(float)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root = "/Users/ezgicakir/Downloads/wogs"
    filenames = os.listdir(root)

    for file in filenames:
        if 'prm' not in file and 'gt' not in file and 'npy' in file:
            orig_pc_file = open (os.path.join(root_dir, data_train_path, file.replace('npy', 'bin')), 'rb')
            orig_pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
            crm = np.load( os.path.join(root_dir, crm_train_path_pc, file)).astype(float)

            calibs = Calibration( os.path.join(root_dir, calib_train_path), file.replace('npy', 'txt'))

            gt_label_file = os.path.join(root_dir, labels_path, file.replace('npy', 'txt'))
            #gt_label_file = os.path.join('/Users/ezgicakir/Desktop/pvrcnn_kitti_preds/pvrcnn_kitti_preds_full/val', file.replace('npy', 'txt'))
            pred_label_file = os.path.join(root, file.replace('npy', 'txt'))

            gt_lines = read_labels( gt_label_file)
            gt_lines = read_car_labels(gt_label_file)
            pred_lines = read_labels( pred_label_file)

            print("GT BOXES")
            gt_bboxes = get_bboxes(labels=gt_lines, calibs=calibs)
            print("PRED BOXES")
            pred_bboxes = get_bboxes(labels=pred_lines, calibs=calibs)

            out = np.load( os.path.join(root, file))#.astype(float)

            import math
            pc = out[:,0:4]
            print(np.min(pc[:,0]), np.max(pc[:,0]))
            out = out[:,-1].reshape(-1,1) + 1e-10
            out *= (np.log(pc[:,0])).reshape(-1,1) #(np.log(pc[:,0])/math.log(5,10))
            print("file: ", file)
            print("output limits: ", np.max(out), np.min(out))
            #out = generate_prm_mask(out)

            #Preds are in green, ground truth are in blue
            visualize_pointcloud( pc, out, pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes, mult=1, idx=0)
            #visualize_pointcloud_onlypreds( pc, out, pred_bboxes=pred_bboxes, mult=1, idx=0)
            for i in range(110):
                prm_file = os.path.join(root, file.replace('.npy', f'_prm_{i}.npy'))
                if not os.path.exists(prm_file):
                    continue
                prm = np.load(prm_file).astype(float)
                #print("prm limits: ", np.max(prm[:,0]), np.min(prm[:,0]))
                #print("center location: ", pc[np.argmax(prm)], np.argmax(prm))
                prm = prm[:,0]
                prm_ = generate_prm_mask(prm)

                visualize_pointcloud( pc, prm_, pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes , idx=0, mult=10000000)
                #visualize_prm(pc, prm_, bboxes=bboxes)

