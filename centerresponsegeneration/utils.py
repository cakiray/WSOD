import os
import numpy as np
import struct
import open3d
import matplotlib.pyplot as plt

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

def read_labels( label_path, idx=None):
    if idx is not None:
        path =os.path.join(label_path, '%06d.txt' % idx)
    else:
        path = label_path
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

def visualize_pointcloud( pc, colors, bboxes=None, idx=0, mult= 1, number= 0):

    colors = mult* colors
    color = np.zeros((colors.shape[0], 3))
    if len(colors.shape) == 2:
        colors = colors[:,0]

    color[:,idx] = colors
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(pc[:, 0:3])
    pcd.colors = open3d.open3d.utility.Vector3dVector(color)
    # for us, it corresponds to class response map

    if bboxes is not None:
        # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                 [4, 5], [5, 6], [6, 7], [4, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        lenght = len(lines)
        for i in range(len(bboxes)//8-1): #for multiple lines, need to extend lines
            newlines = [[0,0] for _ in range(lenght)]
            for j in range(lenght):
                newlines[j][0] = lines[j][0] + (8 * (i+1))
                newlines[j][1] = lines[j][1] + (8 * (i+1))
            # newlines will span 8 to 9, 9 to 10, so on...
            lines.extend(newlines)
        # Use the same color for all lines
        clrs = [[0, 1, 0] for _ in range(len(lines))]

        line_set = open3d.open3d.geometry.LineSet()
        line_set.points = open3d.open3d.utility.Vector3dVector(bboxes)
        line_set.lines = open3d.open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.open3d.utility.Vector3dVector(clrs)
        open3d.open3d.visualization.draw_geometries([pcd,line_set])
    else:
        open3d.open3d.visualization.draw_geometries([pcd])

    """
    # To save the visualized point cloud
    vis = open3d.open3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("/Users/ezgicakir/Downloads/outputs-3/%s" % number)
    vis.destroy_window()
    """

def distance( p1, p2, _in3d=True):
    if _in3d:
        return np.sqrt( (p1[:,0]-p2[0])**2 +  (p1[:,1]-p2[1])**2 + (p1[:,2]-p2[2])**2 )
    else:
        return np.sqrt( (p1[:,0]-p2[0])**2 + (p1[:,1]-p2[1])**2 )

# calculates the normalized value of object in point cloud
def set_normal_label(max, min, point, filt , _in3d=True):
    
    labels =  1 - ( distance(point[:,0:3] , min, _in3d) / distance(np.array([max]),min, _in3d) )
    return labels[filt]

def get_crm_of_object(points, center, threshold, _in3d = False):
    labels =  distance(points[:,0:3] , center, _in3d) # calculate distance of points to center point
    #labels[labels>=threshold] = 0 # discard the points that are away than threshold, meaning no longer belong to a object

    labels_normalized = labels / threshold # normalize labels

    labels_normalized = 1- labels_normalized # make center to be 1 and values decrease while getting away
    labels_normalized[labels_normalized<0] = 0
    return labels_normalized.reshape(-1,1)
