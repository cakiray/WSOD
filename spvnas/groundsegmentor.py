import argparse
import numpy as np
import open3d
import os
#from torchpack.utils.config import configs
import centerresponsegeneration.utils as utils


planes_model =  open('/Users/ezgicakir/Downloads/planes/000000.txt', 'r').readlines()[3].rstrip()

a,b,c,d = map(float, planes_model.split(' '))

if b > 0:
    a,b,c,d = -a, -b, -c , -d

print(f"plane model : {a}, {b},{c},{d}, {type(d)}")
block_ = np.fromfile('/Users/ezgicakir/Documents/Thesis/data/data_object_velodyne/training/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4)#[:,0:3]
labels_ = np.load( '/Users/ezgicakir/Documents/Thesis/data/center_response_map_r2/000000.npy').astype(float)
mask = np.ones(block_.shape[0], dtype=bool)
plane_eq = a*block_[:,0] + b*block_[:,1] + c*block_[:,2] + d

pcd=open3d.open3d.geometry.PointCloud()
pcd.points= open3d.open3d.utility.Vector3dVector(block_[:, 0:3])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.15,
                                         ransac_n=100,
                                         num_iterations=10000)

eps = 1e-1
print(plane_eq)
print(np.any(plane_eq==0))
mask_and = np.logical_and(plane_eq<=eps, plane_eq>=-eps)
#mask[mask_and] = False

mask[plane_eq<=eps] = False
block = block_[mask]
labels = labels_[mask]

utils.visualize_pointcloud( block, labels,  mult= 1, idx=0)

mask_ = np.ones(block_.shape[0], dtype=bool)
mask_[inliers] = False
block = block_[mask_]
labels = labels_[mask_]

utils.visualize_pointcloud( block, labels,  mult= 1, idx=0)
exit(0)
parser = argparse.ArgumentParser()
parser.add_argument('configs', metavar='FILE', help='configs file path')
args, opts = parser.parse_known_args()
configs.load(args.configs, recursive=True)
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

for i in range(len(pcsfiles)):
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

