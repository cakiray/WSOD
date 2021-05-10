import os
import numpy as np
import struct
import open3d

def load_pc(f):
    file = open(f, 'rb')
    points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:,0:3]
    return points

def read_bin_velodyne(path, idx=None):

    pc_list=[]
    with open( os.path.join(path,'%06d.bin' % idx),'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])

    return np.asarray(pc_list,dtype=np.float32)

def read_labels( label_path, idx):

    path =os.path.join(label_path, '%06d.txt' % idx)
    label = np.loadtxt(path,
                       dtype={'names': ('type', 'truncated', 'occuluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'h', 'w', 'l', 'x', 'y', 'z','rotation_y'),
                              'formats': ('S14', 'float', 'float', 'float', 'float', 'float', 'float', 'float','float', 'float', 'float', 'float', 'float', 'float', 'float')})

    if label.size == 1:
        label = label[np.newaxis]

    return label

def read_points( lidar_path, idx):

    path = os.path.join(lidar_path, '%06d.bin' % idx)
    file = open(path, 'rb')
    points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:,0:3]  # .astype(np.float16)
    return points

def visualize_pointcloud( pc, colors, idx):
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(pc[:, 0:3])
    colors = 255* colors
    color = np.zeros((colors.shape[0], 3))
    if len(colors.shape) == 2:
        colors = colors[:,0]

    color[:,idx+1] = colors

    pcd.colors = open3d.open3d.utility.Vector3dVector(color) # for us, it corresponds to class response map

    open3d.open3d.visualization.draw_geometries([pcd])

def distance( p1, p2, _in3d=True):
    if _in3d:
        return np.sqrt( (p1[:,0]-p2[0])**2 +  (p1[:,1]-p2[1])**2 + (p1[:,2]-p2[2])**2 )
    else:
        return np.sqrt( (p1[:,0]-p2[0])**2 + (p1[:,1]-p2[1])**2 )

# calculates the normalized value of object in point cloud
def set_normal_label(max, min, point, filt , _in3d=True):
    
    labels =  1 - ( distance(point[:,0:3] , min, _in3d) / distance(np.array([max]),min, _in3d) )
  
    return labels[filt]

