# This is a sample Python script.
from centerresponsegeneration.utils import *
import torch
import numpy as np
from centerresponsegeneration.config import *
from centerresponsegeneration.utils import *
from centerresponsegeneration.calibration import Calibration
from spvnas.core.utils import *
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    root = "/Users/ezgicakir/Downloads/5epo_spvnas"
    filenames = os.listdir(root)

    for file in filenames:
        if 'prm' not in file and 'gt' not in file and 'npy' in file:
            orig_pc_file = open (os.path.join(root_dir, data_train_path, file.replace('npy', 'bin')), 'rb')
            orig_pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
            crm = np.load( os.path.join(root_dir, crm_train_path_pc, file)).astype(float)

            label_file = os.path.join(root_dir, labels_path, file.replace('npy', 'txt'))
            calibs = Calibration( os.path.join(root_dir, calib_train_path), file.replace('npy', 'txt'))
            lines = read_labels( label_file)

            out = np.load( os.path.join(root, file)).astype(float)
            pc = out[:,0:4]
            out = out[:,-1].reshape(-1,1)

            bboxes = get_bboxes(labels=lines, calibs=calibs)
            print("file: ", file)
            print("output limits: ", np.max(out), np.min(out))
            #out = generate_prm_mask(out)
            visualize_pointcloud( pc, out, bboxes, mult=5, idx=0)

            for i in range(-10):
                prm_file = os.path.join(root, file.replace('.npy', f'_prm_{i}.npy'))
                if not os.path.exists(prm_file):
                    break
                prm = np.load(prm_file).astype(float)
                print("prm limits: ", (prm[:,0]>0.0).sum(0), np.max(prm[:,0]), np.min(prm[:,0]))
                prm = prm[:,0]
                #mask = prm<0.00005
                #prm[mask] =0.0
                prm_ = generate_prm_mask(prm)
                #visualize_pointcloud( pc, prm_, bboxes , idx=0, mult=5)
                visualize_prm(pc, prm_, bboxes=bboxes)