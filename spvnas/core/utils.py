import math
import os
import numpy as np
import struct
import open3d
import torch
import torchvision as tv

__all__ = [ 'load_pc', 'read_bin_velodyne', 'read_labels' , 'read_points' , 'get_bboxes', 'box_center_to_corner',
            'generate_car_masks',  'generate_prm_mask', 'FPS', 'KNN']

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

def get_car_num(label):
    count = 0
    for data in label:
        if data['type'] == b'Car':
            count += 1
    return count

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

def find_bbox(point, labels, calibs, class_='Car'):
    bboxes = get_bboxes(labels=labels, calibs=calibs)
    i=-1
    for label in labels:
        i += 1
        if label['type'] == b'Car':
            bbox = bboxes[i*8:(i+1)*8-1]
            p1, p2, p4, p5 = bbox[0], bbox[1], bbox[3], bbox[4]
            u, v, w, p = p2-p1, p4-p1, p5-p1, point-p1
            ux = np.dot(p,u) < np.dot(u,u) and np.dot(p,u)>0.0 
            vx = np.dot(p,v) < np.dot(v,v) and np.dot(p,v)>0.0 
            wx = np.dot(p,w) < np.dot(w,w) and np.dot(p,w)>0.0 
        
            if (ux and vx) and wx:
                return np.asarray([label]), i

    return None, -1

# Return TP and FN number of peaks detected for car
def tp_fn_peak(labels, bbox_found):
    tp = 0
    fn = 0
    for i,label in enumerate(labels):
        if label['type'] == b'Car':
            if bbox_found[i] >0:
                tp += 1
            else:
                fn += 1
    return tp, fn

def generate_car_masks(points, labels, calibs):
    points = np.asarray(points)
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
    mask = np.zeros_like(prm)
    mask[prm > 0.0] = 1
    mask[prm <= 0.0] = 0
    return mask

def normalize(arr):
    min = np.min(arr[arr>0.0])
    max = np.max(arr)
    arr = (arr-min)/(max-min)
    arr[arr<0.0] = 0.0
    return arr

# Segments ground of predictions
def segment_ground(points, preds, distance_threshold=0.15):
    points = points.detach().cpu().clone()
    pcd=open3d.open3d.geometry.PointCloud()
    
    pcd.points= open3d.open3d.utility.Vector3dVector(points[:, 0:3])
    #inlers contains the indexes of ground points
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=100,
                                             num_iterations=1000)
    preds[inliers] = 0.0

    return preds

def FPS(peaks_idxs, points,  num_frags=-1):
    from scipy import spatial
    """Fragmentation by the furthest point sampling algorithm.
    The fragment centers are found by iterative selection of the vertex from
    vertices that is furthest from the already selected vertices. The algorithm
    starts with the centroid of the object model which is then discarded from the
    final set of fragment centers.
    A fragment is defined by a set of points on the object model that are the
    closest to the fragment center.
    Args:
      peaks: [num_vertices, 3] ndarray with 3D vertices of the object model.
      num_frags: Number of fragments to define.
    Returns:
      [num_frags, 3] ndarray with fragment centers and [num_vertices] ndarray
      storing for each vertex the ID of the assigned fragment.
    """
    if num_frags == -1:
        num_frags =max(3, len(peaks_idxs)//10)
    
    # Start with the origin of the model coordinate system.
    peak_centers = [np.array([0., 0., 0.])]
    peak_locs = [] # 3D location of peaks
    valid_peak_list = []
    valid_indexes = []
    for i in range(len(peaks_idxs)):
        peak_ind = peaks_idxs[i][2]  #peak_list: [0,0,indx]
        peak_locs.append( points[peak_ind] )

    # Calculate distances to the center from all the vertices.
    nn_index = spatial.cKDTree(peak_centers)
    nn_dists, _ = nn_index.query(peak_locs, k=1)

    for _ in range(num_frags):
        # Select the furthest vertex as the next center.
        new_center_ind = np.argmax(nn_dists)
        new_center = peak_locs[new_center_ind]
        peak_centers.append(peak_locs[new_center_ind])
        valid_peak_list.append(peaks_idxs[new_center_ind])
        valid_indexes.append(new_center_ind)
        # Update the distances to the nearest center.
        nn_dists[new_center_ind] = -1
        nn_dists = np.minimum(
            nn_dists, np.linalg.norm(peak_locs - new_center, axis=1))

    # Remove the origin.
    peak_centers.pop(0)# 3D info of peak centers

    new_peak_centers = []
    new_valid_peak_list = []
    new_valid_indexes = []

    """
    #Eliminate points close each other more that 2.5 mt
    for i,center in enumerate(peak_centers):
        far = True
        for new_center in new_peak_centers:
            dist = L2dist_2d(center, new_center)
            if dist < 2.5:
                far=False
                break
                
        if far:
            new_peak_centers.append(center)
            new_valid_peak_list.append(valid_peak_list[i])
            new_valid_indexes.append(valid_indexes[i])
    """
    return np.asarray(new_peak_centers), np.asarray(new_valid_peak_list), np.asarray(new_valid_indexes)

def L2dist_2d(p1, p2):
    return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def KNN(points, anchor, k=10):
    points = points.detach().cpu().clone()
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(points[:, 0:3])
    pcd_tree = open3d.geometry.KDTreeFlann(pcd)

    [k, idxs, _] = pcd_tree.search_knn_vector_3d(pcd.points[anchor], k)

    return idxs

#https://github.com/sshaoshuai/PointRCNN/blob/1d0dee91262b970f460135252049112d80259ca0/tools/eval_rcnn.py
#KITTI format of bounding box of instances
def get_kitti_format( points, crm, peak_list, peak_responses, calibs) :
    bboxs_raw = []
    for i,response in enumerate(peak_responses):
        #dimension are of anchor
        h,w,l = 1.52563191462, 1.62856739989, 3.88311640418

        mask = response.flatten()>0.0
        obj_mask = points[mask][:,0:3]
        if obj_mask.shape[0]<4:
            continue
        
        pc_ = open3d.utility.Vector3dVector(obj_mask)
        bbox = open3d.geometry.AxisAlignedBoundingBox()
        bbox = bbox.create_from_points(pc_)
        """
        # To calculate rotation around z(up):
        # Set z=0 and z=1 for all points in the object mask
        # Calculate rotation matrix R, orthogonal unit-vectors pointing on rotated x, y and z directions
        # Rotation around y is given by arctan2(y,x)
        bbox_oriented = open3d.geometry.OrientedBoundingBox()
        size = obj_mask.shape[0]
        pc_or = np.zeros(shape=(size*2,3))
        pc_or[:size] = obj_mask
        pc_or[size:] = obj_mask
        pc_or[:size, -1] = 0
        pc_or[size:, -1] = 1
        
        pc_or = open3d.utility.Vector3dVector(pc_or)
        bbox_oriented = bbox_oriented.create_from_points(pc_or)
        #bbox_oriented.extent # extension of convex hull on x,y,z
        R = bbox_oriented.R
        if  np.linalg.det(R) < 0:
           R = -R

        #orientation_vector = np.arctan2( R[:,1], R[:,0])  # (3,1) vector as (ry, ry+pi/2, 0) since z doesn't have rotation
        #ry = orientation_vector[0]
        from pyquaternion import Quaternion
        quat = Quaternion(matrix=R)
        ry = quat.radians #+ np.pi/2
        if quat.get_axis(undefined=[0,0,0])[2] == -1:
            ry += np.pi/2
        else:
            ry -= np.pi/2
        """
        #get center of bbox and convert from velo to rect
        np_center = bbox.get_center().reshape(1,3) #numpy, 1x3, in velo
        #np_center = bbox_oriented.get_center().reshape(1,3)
        np_center = calibs.project_velo_to_rect(np_center) # x,y,z in velo -> z,x,y in rect

        
        rect, R, _, _ = _rectangle_search(x=obj_mask[:,0], y=obj_mask[:,1])
        
        from pyquaternion import Quaternion
        quat = Quaternion(matrix=R)

        ry = quat.radians + np.pi/2
        #np_center = calibs.project_velo_to_rect(center)
        
        """
        corners_o3d = bbox.get_box_points() #open3d.utility.Vector3dVector
        np_corners = np.asarray(corners_o3d) #Numpy array, 8x3
        #corners from velodyne to rect
        np_corners = calibs.project_velo_to_rect(np_corners) # 8x3
        
        # h->z, w->x, l->y
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        dimensions = max_bound-min_bound #length of each side of bounding box, eg. x,y,z:w,l,h
        w, l, h = dimensions[0], dimensions[1], dimensions[2] # in velodyne coord order
        
        """
        #3D bounding box's corners location on image
        np_corners = get_corners(np_center) # in rect coord
        corners_img = calibs.corners3d_to_img_boxes(np.asarray([np_corners])) # 1x4

        x, y, z = np_center[0,0], np_center[0,1]+h/2, np_center[0,2] # in rect coord

        #ry = np.pi/2 # pi/2 if ry is not calculated
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry
        score = (crm[peak_list[i][2]].item()+ response[peak_list[i][2]] )/2 # np.log( points[peak_list[i][2],0])
        # kitti format is
        #  type of object 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'
        #  truncated
        #  occluded     0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
        #  alpha        Observation angle of object, ranging [-pi..pi]
        #  bbox         2D bounding box of object in the image, x1,y1,x2,y2 (left,top,right,bottom)
        #  dimensions   3D object dimensions: height, width, length (in meters)
        #  location     3D object location x,y,z in camera coordinates (in meters)
        #  rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        #  score
        bboxs_raw.append(('Car', alpha, corners_img[0,0], corners_img[0,1], corners_img[0, 2], corners_img[0, 3],
                         h, w, l, x, y, z, ry, score))

    return bboxs_raw

def get_corners(center): #center is numpy array in rect coords, [x,y,z]
    #average dimensions of a car
    h,w,l = 1.52563191462, 1.62856739989, 3.88311640418
    x,y,z = center[0,0], center[0,1], center[0,2]
    corners = np.asarray([[x+h/2, y+w/2, z-l/2],
                        [x+h/2, y+w/2, z+l/2],
                        [x-h/2, y+w/2, z-l/2],
                        [x+h/2, y-w/2, z-l/2],
                        [x-h/2, y-w/2, z+l/2],
                        [x-h/2, y-w/2, z-l/2],
                        [x+h/2, y-w/2, z+l/2],
                        [x-h/2, y+w/2, z+l/2]])
    return corners

def nms_torchvision(bboxes_raw):
    box_info = torch.zeros((len(bboxes_raw),4))
    scores = torch.zeros((len(bboxes_raw)))
    for i,bbox in enumerate(bboxes_raw):
        box_info[i,0] = bbox[2]# x1
        box_info[i,1] = bbox[3]# x2
        box_info[i,2] = bbox[4]# y1
        box_info[i,3] = bbox[5]# y2
        #0<=x1<x2, 0<=y1<y2
        scores[i] = bbox[-1]
    
    box_info = box_info.to(device = torch.cuda.current_device())
    scores = scores.to(device = torch.cuda.current_device())
    kept_idxs = tv.ops.nms(boxes=box_info, scores=scores, iou_threshold=0.3 )
    
    return kept_idxs

def save_in_kitti_format(file_id, kitti_output, points, crm, peak_list, peak_responses, calibs, labels):

    kitti_output_file = os.path.join(kitti_output, f'{file_id}.txt')
    kitti_format_list = get_kitti_format(points, crm, peak_list, peak_responses, calibs)
    #kept_idxs = non_maximum_supression(kitti_format_list)
    kept_idxs = nms_torchvision(kitti_format_list)
    with open(kitti_output_file, 'w') as f:
        for idx in kept_idxs:
            kitti_format = kitti_format_list[idx]
            print('%s 0.0 3 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  kitti_format, file=f)

            """img_boxes_w = corners_img[:, 2] - corners_img[:, 0]
           img_boxes_h = corners_img[:, 3] - corners_img[:, 1]
           """
    return kept_idxs

#https://github.com/Silas-Asamoah/Lshape-fitting/tree/f2fc4e52d8c0e7203c36b3fcf4a9d50fa7132003
def _rectangle_search( x, y):

    X = np.array([x, y]).T
    dtheta_deg_for_serarch = 1.0
    criteria = 2

    dtheta = np.deg2rad(dtheta_deg_for_serarch)
    minp = (-float('inf'), None)
    for theta in np.arange(0.0, np.pi / 2.0 - dtheta, dtheta):

        e1 = np.array([np.cos(theta), np.sin(theta)])
        e2 = np.array([-np.sin(theta), np.cos(theta)])

        c1 = X @ e1.T
        c2 = X @ e2.T

        # Select criteria
        if criteria == 1:
            cost = _calc_area_criterion(c1, c2)
        elif criteria == 2:
            cost = _calc_closeness_criterion(c1, c2)
        elif criteria == 3:
            cost = _calc_variance_criterion(c1, c2)

        if minp[0] < cost:
            minp = (cost, theta)

    # calculate best rectangle
    sin_s = np.sin(minp[1])
    cos_s = np.cos(minp[1])

    c1_s = X @ np.array([cos_s, sin_s]).T
    c2_s = X @ np.array([-sin_s, cos_s]).T

    rect = {'a': [], 'b': [], 'c': []}
    rect['a'].append(cos_s)
    rect['b'].append(sin_s)
    rect['c'].append(min(c1_s))

    rect['a'].append(-sin_s)
    rect['b'].append(cos_s)
    rect['c'].append(min(c2_s))

    rect['a'].append(cos_s)
    rect['b'].append(sin_s)
    rect['c'].append(max(c1_s))

    rect['a'].append(-sin_s)
    rect['b'].append(cos_s)
    rect['c'].append(max(c2_s))

    R = np.asarray([[cos_s, sin_s, 0],
                    [-sin_s, cos_s, 0],
                    [0, 0, 1]])

    c1 = calc_cross_point(a=rect['a'][0:2], b=rect['b'][0:2], c=rect['c'][0:2])
    c2 = calc_cross_point(a=rect['a'][1:3], b=rect['b'][1:3], c=rect['c'][1:3])
    c3 = calc_cross_point(a=rect['a'][2:4], b=rect['b'][2:4], c=rect['c'][2:4])
    c4 = calc_cross_point(a=[rect['a'][0],rect['a'][3]], b=[rect['b'][0],rect['b'][3]], c=[rect['c'][0],rect['c'][3]])
    corners_velo = np.asarray([c1,c2,c3,c4])

    center = np.asarray( [(c1[0]+c3[0])/2, (c1[1]+c3[1])/2])

    return rect, R, center, corners_velo


def calc_cross_point( a, b, c):
    x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
    y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
    return x, y, 0 # 0 is as 3rd direction corner


def _calc_area_criterion(c1, c2):
    c1_max = max(c1)
    c2_max = max(c2)
    c1_min = min(c1)
    c2_min = min(c2)

    alpha = -(c1_max - c1_min) * (c2_max - c2_min)

    return alpha

def _calc_closeness_criterion( c1, c2):
    min_dist_of_closeness_crit = 0.01

    c1_max = max(c1)
    c2_max = max(c2)
    c1_min = min(c1)
    c2_min = min(c2)

    D1 = [min([np.linalg.norm(c1_max - ic1),
               np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
    D2 = [min([np.linalg.norm(c2_max - ic2),
               np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]

    beta = 0
    for i, _ in enumerate(D1):
        d = max(min([D1[i], D2[i]]), min_dist_of_closeness_crit)
        beta += (1.0 / d)

    return beta

def _calc_variance_criterion( c1, c2):
    c1_max = max(c1)
    c2_max = max(c2)
    c1_min = min(c1)
    c2_min = min(c2)

    D1 = [min([np.linalg.norm(c1_max - ic1),
               np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
    D2 = [min([np.linalg.norm(c2_max - ic2),
               np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]

    E1, E2 = [], []
    for (d1, d2) in zip(D1, D2):
        if d1 < d2:
            E1.append(d1)
        else:
            E2.append(d2)

    V1 = 0.0
    if E1:
        V1 = - np.var(E1)

    V2 = 0.0
    if E2:
        V2 = - np.var(E2)

    gamma = V1 + V2

    return gamma
