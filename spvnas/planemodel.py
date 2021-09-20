import os
import open3d
import numpy as np

root = '/data/dataset/kitti/object/training/velodyne'
planesroot = '/data/Ezgi/plane_models'
filenames = os.listdir(root)

for file in filenames:
    block_ = np.fromfile(os.path.join(root, file), dtype=np.float32).reshape(-1, 4)#[:,0:3]
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(block_[:, 0:3])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                             ransac_n=100,
                                             num_iterations=1000)

    file1 = open(os.path.join(planesroot, file.replace('bin', 'txt')),"w")
    # \n is placed to indicate EOL (End of Line)
    for x in plane_model:
        file1.write(str(x))
        file1.write(' ')
    file1.close()