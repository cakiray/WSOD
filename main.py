# This is a sample Python script.
from centerresponsegeneration.utils import visualize_pointcloud, read_car_labels
import os
import torch
import numpy as np
from centerresponsegeneration.config import *
from centerresponsegeneration.calibration import Calibration
#from spvnas.core.datasets.utils import *
from centerresponsegeneration.utils import *
if __name__ == '__main__':

    recalls = np.asarray([0.30, 0.41, 0.66, 0.75, 0.82, 0.0])
    precisions = np.asarray( [0.80, 0.76, 0.66, 0.60, 0.45, 1.0])

    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])*100
    f1 = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))


    import pickle
    file = '/Users/ezgicakir/Downloads/ex/result.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    #print(data[0])
    data = data[18]
    sample = data['frame_id']
    print(sample)
    kitti_labels = create_kitti_format_annot_pkl(data, only_car=True)
    
    #sample ='004578'
    orig_pc_file = open ('/Users/ezgicakir/Documents/Thesis/data/training/velodyne/'+sample+'.bin', 'rb')
    orig_pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
    calibs = Calibration( '/Users/ezgicakir/Documents/Thesis/data/training/calib/', sample+'.txt')
    pred_bboxes = get_bboxes(labels=kitti_labels, calibs=calibs)

    gt_label_file = f'/Users/ezgicakir/Documents/Thesis/data/training/label_2/{sample}.txt'
    gt_lines = read_car_labels( gt_label_file)
    gt_bboxes = get_bboxes(labels=gt_lines, calibs=calibs)
    out = np.zeros( (orig_pc.shape[0],1))
    visualize_pointcloud( orig_pc, out, pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes,  mult=0, idx=0)
    exit(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root = "/Users/ezgicakir/Downloads/a"
    filenames = os.listdir(root)
    filenames = sorted(filenames, reverse=True)

    for file in filenames:
        sample = file[0:6]

        if '_' not in file and 'crm' not in file and 'npy' in file:
            #sample = '004567'
            orig_pc_file = open ('/Users/ezgicakir/Documents/Thesis/data/training/velodyne/'+sample+'.bin', 'rb')
            orig_pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]


            #crm = np.load( os.path.join(root_dir, crm_train_path_pc, file)).astype(float)
            calibs = Calibration( '/Users/ezgicakir/Documents/Thesis/data/training/calib/', sample+'.txt')
            out = np.load( os.path.join(root, file))#.astype(float)
            print(sample, np.min(out), np.max(out))
            #visualize_pointcloud( orig_pc, out, mult=1, idx=0)
            #continue

            gt_label_file = os.path.join(root_dir, labels_path, file.replace('npy', 'txt'))
            gt_label_file = f'/Users/ezgicakir/Documents/Thesis/data/training/label_2/{sample}.txt'
            gt_lines = read_labels( gt_label_file)
            gt_bboxes = get_bboxes(labels=gt_lines, calibs=calibs)

            pred_label_file = f'/Users/ezgicakir/Documents/Thesis/data/training/label_2/{sample}.txt'
            pred_lines = read_labels( pred_label_file)
            pred_bboxes = get_bboxes(labels=pred_lines, calibs=calibs)
            gt_bboxes = pred_bboxes
            #visualize_pointcloud( orig_pc, out,pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes,  mult=2, idx=0)
            continue
            #gt_label_file = os.path.join('/Users/ezgicakir/Desktop/pvrcnn_kitti_preds/pvrcnn_kitti_preds_full/val', file.replace('npy', 'txt'))
            pred_label_file = os.path.join(root, file.replace('npy', 'txt'))


            out = np.load( os.path.join(root, file))#.astype(float)

            import math
            pc = out[:,0:4]
            #out = generate_prm_mask(out)

            #Preds are in green, ground truth are in blue
            visualize_pointcloud( pc, out, pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes, mult=1, idx=0)
            #visualize_pointcloud_onlypreds( pc, out, pred_bboxes=pred_bboxes, mult=1, idx=0)
            for i in range(-110):
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


        else:
            calibs = Calibration( '/Users/ezgicakir/Documents/Thesis/data/training/calib/', sample+'.txt')

            gt_label_file = os.path.join(root_dir, labels_path, file.replace('npy', 'txt'))
            gt_label_file = f'/Users/ezgicakir/Documents/Thesis/data/training/label_2/{sample}.txt'
            gt_lines = read_labels( gt_label_file)
            gt_bboxes = get_bboxes(labels=gt_lines, calibs=calibs)

            #pred_label_file =  f'/Users/ezgicakir/Documents/Thesis/data/training/label_2/{sample}.txt'
            pred_label_file =  f'/Users/ezgicakir/Downloads/a/{sample}.txt'
            pred_lines = read_labels( pred_label_file)
            print(pred_lines)
            pred_bboxes = get_bboxes(labels=pred_lines, calibs=calibs)

            print("else")
            print(file)
            out = np.load( os.path.join(root, file))#.astype(float)
            #out[out<0.5 ] = 0
            print(out[:,-1].shape, np.min(out[:,-1]), np.max(out[:,-1]))
            o = out[:,-1]
            o[o>0]=1
            #visualize_pointcloud_generatebox(out[:,0:3], o)
            visualize_pointcloud( out[:,0:3], o, gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, mult=1, idx=0)
