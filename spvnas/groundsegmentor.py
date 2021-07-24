import argparse
import numpy as np
import open3d
import os
from torchpack.utils.config import configs

parser = argparse.ArgumentParser()
parser.add_argument('configs', metavar='configs', help='configs file path')
args, opts = parser.parse_known_args()
configs.load(args.config, recursive=True)
configs.update(opts)

pcs_root = os.path.join(configs.dataset.root, configs.dataset.data_path)
crm_pcs_root = os.path.join(configs.dataset.root,configs.dataset.crm_path)

pcs_gs = pcs_root+'_gs'
crm_pcs_gs = crm_pcs_root+'_gs'

if not os.path.exists(pcs_gs):
    os.mkdir(pcs_gs)
if not os.path.exists(crm_pcs_gs):
    os.mkdir(crm_pcs_gs)

pcsfiles = os.listdir(pcs_root)
crmfiles = os.listdir(crm_pcs_root)
pcsfiles.sort()
crmfiles.sort()

for i in len(pcsfiles):
    pc_file = open ( os.path.join(pcs_root,pcsfiles[i]), 'rb')
    block_ = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
    labels_ = np.load( os.path.join(crm_pcs_root,crmfiles[i] )).astype(float)

    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(block_[:, 0:3])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.15,
                                             ransac_n=100,
                                             num_iterations=10000)

    mask = np.ones(block_.shape[0], dtype=bool)
    mask[inliers] = False
    block_ = block_[mask]

    mask = np.ones(labels_.shape[0], dtype=bool)
    mask[inliers] = False
    labels_ = labels_[mask]

    file = open(os.path.join(pcs_gs, pcsfiles[i]), "wb")
    file.write(block_)
    file.close()
    np.save( os.path.join(crm_pcs_gs, crmfiles[i]), labels_)

