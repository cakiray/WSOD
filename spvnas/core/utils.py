import os
import numpy as np
import struct
import open3d

__all__ = [ 'load_pc', 'read_bin_velodyne', 'read_labels' , 'read_points' , 'get_bboxes', 'box_center_to_corner', 'iou', 'generate_car_masks',  'generate_prm_mask']

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
            ry = data['rotation_y'] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            xyz = np.array([[x,y,z]])
            t = (xyz[0][0], xyz[0][1], xyz[0][2])  # location (x,y,z) in camera coord.

            bbox = [h,w,l,t,ry]
            bboxes.extend(box_center_to_corner(bbox, calibs))

    return bboxes

def box_center_to_corner(data, calibs):
    #First rotate and transform in camera position,
    # then convert from camera to velodyne coords.
    translation = data[3]
    h, w, l = data[0], data[1], data[2]
    rotation = data[4]

    # Create a bounding box outline
    bounding_box = np.array([
        [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],
        [0, 0, 0, 0, -h, -h, -h, -h],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        ])

    # Standard 3x3 rotation matrix around the Y axis in camera coords.
    rotation_matrix = np.array([
        [np.cos(rotation),0.0, np.sin(rotation)],
        [0.0, 1.0, 0.0],
        [-np.sin(rotation), 0.0, np.cos(rotation)]])

    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box ).T+ eight_points
    for i in range(len(corner_box)):
        # convert corners to velodyne coords.
        corner_box[i]= calibs.project_rect_to_velo(np.array([corner_box[i]]))
    return corner_box


def iou(preds, targets, n_classes=2):

    ious = np.zeros(n_classes)
    pred = preds.reshape(-1,1)
    target = targets.reshape(-1,1)
    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum()  # Cast to long to prevent overflows

        union = pred_inds.astype(np.int_).sum() + target_inds.astype(np.int_).sum() - intersection
        if union == 0:
            ious[cls] = float('nan')  # If there is no ground truth, do not include in evaluation
        else:
            ious[cls] = float(intersection) / float(max(union, 1))

    return np.array(ious)

def generate_car_masks(points, labels, calibs):
    points = np.asarray(points)
    #points = points[points[:,0]>0]
    bboxes = get_bboxes(labels=labels, calibs=calibs)
    map = np.zeros((points.shape[0], 1))
    i=-1
    for label in labels:
        if label['type'] != b'DontCare':
            i += 1
        if label['type'] == b'Car':
            bbox = bboxes[i*8:(i+1)*8-1]
            obbox = open3d.open3d.geometry.OrientedBoundingBox()
            bboc_vec = open3d.utility.Vector3dVector(bbox)
            obbox = obbox.create_from_points(bboc_vec)
            points_vec = open3d.utility.Vector3dVector(points)
            b_points = obbox.get_point_indices_within_bounding_box(points_vec)

            map[b_points] = 1
    #utils.visualize_pointcloud(points, map, bboxes=bboxes)

    return map

def generate_prm_mask(prm):
    prm[prm > 0.0] = 1
    prm[prm <= 0.0] = 0
    return prm