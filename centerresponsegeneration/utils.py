import os
import numpy as np
import struct
import open3d
from .calibration import *

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

def visualize_prm(pc, prm, bboxes=None):
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(pc[:, 0:3])
    color = np.zeros((prm.shape[0], 3))
    if len(prm.shape) >1 and prm.shape[1]==1:
        color[:,0] = prm[:,0]
    else:
        color[:,0] = prm
    pcd.colors = open3d.open3d.utility.Vector3dVector(color)

    mask = prm>0.0
    pc_ = pc[mask]
    pc_ = open3d.utility.Vector3dVector(pc[mask][:,0:3])

    bbox = open3d.geometry.AxisAlignedBoundingBox()
    bbox.color = [1,0,1]
    bbox = bbox.create_from_points(pc_)

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
        open3d.open3d.visualization.draw_geometries([pcd,line_set, bbox])
    else:
        open3d.open3d.visualization.draw_geometries([pcd, bbox])

def visualize_pointcloud_orig( pc, boxes=None):
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(pc[:, 0:3])
    if boxes is not None:
        # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                 [4, 5], [5, 6], [6, 7], [4, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        lenght = len(lines)
        for i in range(len(boxes)//8-1): #for multiple lines, need to extend lines
            newlines = [[0,0] for _ in range(lenght)]
            for j in range(lenght):
                newlines[j][0] = lines[j][0] + (8 * (i+1))
                newlines[j][1] = lines[j][1] + (8 * (i+1))
            # newlines will span 8 to 9, 9 to 10, so on...
            lines.extend(newlines)
        # Use the same color for all lines
        clrs = [[0, 0, 0] for _ in range(len(lines))]

        line_set = open3d.open3d.geometry.LineSet()

        line_set.points = open3d.open3d.utility.Vector3dVector(boxes)
        line_set.lines = open3d.open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.open3d.utility.Vector3dVector(clrs)
        open3d.open3d.visualization.draw_geometries([pcd,line_set])
    else:
        open3d.open3d.visualization.draw_geometries([pcd])

def visualize_pointcloud( pc, colors, bboxes=None, idx=0, mult= 1, number= 0):

    colors = mult* colors
    color = np.zeros((colors.shape[0], 3))
    if len(colors.shape) >1 and colors.shape[1]==1:
        color[:,idx] = colors[:,0]
    elif len(colors.shape) >1 and colors.shape[1]>1:
        color = colors[:,0:3]
    else:
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


def distance( p1, p2, _in3d=False):
    if _in3d:
        return np.sqrt( (p1[:,0]-p2[0])**2 +  (p1[:,1]-p2[1])**2 + (p1[:,2]-p2[2])**2 ).astype('float')
    else:
        dist =  np.sqrt( (p1[:,0]-p2[0])**2 + (p1[:,1]-p2[1])**2 ).astype('float')
        return dist

# calculates the normalized value of object in point cloud
def set_normal_label(max, min, point, filt , _in3d=False):
    labels =  1 - ( distance(point[:,0:3] , min, _in3d) / distance(np.array([max]),min, _in3d) )
    return labels[filt]

def get_crm_of_object(points, center, _in3d = False):
    labels =  distance(points[:,0:3] , center, _in3d) # calculate distance of points to center point
    #labels[labels>=threshold] = 0 # discard the points that are away than threshold, meaning no longer belong to a object
    return labels

def standardize(data, threshold):
    labels_normalized = data / threshold # normalize labels
    return labels_normalized

def gaussian_distance(data, c=3):
    data = (data/c)**2
    data =  np.exp( -data )

    return data.reshape(-1,1)

def take_power(data, threshold= 1.5, power=3):
    data[data<threshold] = data[data<threshold]**power
    return data

def weight_data(data):
    weights=[0.2, 0.3, 0.7, 0.8, 1]
    pivot = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in range(len(pivot)-1):
        mask = np.logical_and(data>=pivot[i], data<=pivot[i+1])
        data[mask] = data[mask] * weights[i]
    return data

def substact_1(data):
    data = 1- data # make center to be 1 and values decrease while getting away
    data[data<0] = 0
    return data.reshape(-1,1)


def get_bboxes( labels, calibs):
    bboxes = []
    for data in labels:
        if data['type'] != b'DontCare':
            h = data['h'] # box height
            w = data['w'] # box width
            l = data['l']  # box length (in meters)
            x = data['x']
            y = data['y']
            z = data['z']
            ry = data['rotation_y']
            alpha = data['alpha']
            xyz = calibs.project_rect_to_velo(np.array([[x,y,z]]))

            t = (xyz[0][0], xyz[0][1], xyz[0][2])  # location (x,y,z) in camera coord.
            # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            bbox = [h,w,l,t,ry]
            bboxes.extend(box_center_to_corner(bbox,alpha))

    return bboxes

def box_center_to_corner(data,alpha):
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = data[3]
    h, w, l = data[0], data[1], data[2]
    rotation = data[4]
    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0, 0, 0, 0, h, h, h, h]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])
    angle = -1*(rotation+alpha)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]])
    print("rot, alpha" , rotation, alpha)
    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    #corner_box = bounding_box + eight_points.transpose()
    return corner_box.transpose()
