import os
import numpy as np
import struct
import open3d
import torch 

__all__ = [ 'load_pc', 'read_bin_velodyne', 'read_labels' , 'read_points' , 'get_bboxes', 'box_center_to_corner', 'iou',
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

def iou_precision(peak, points, preds, labels, calibs, n_classes=2):
    precision = np.zeros(n_classes)
    peak = points[peak[2]]
    bbox_label, bbox_idx = find_bbox(peak, labels, calibs)
    if bbox_label:
        prm_target = generate_car_masks(points, bbox_label, calibs).reshape(-1)
    else:
        prm_target = np.zeros_like(preds)
    for cls in range(n_classes):
        preds_inds = preds==cls
        target_inds = prm_target==cls
        tp = (preds_inds[target_inds]).sum()
        fp = preds_inds.sum()-tp
        precision[cls] = float(tp / (fp+tp))

    return np.array(precision)


def iou_recall(peak, points, preds, labels, calibs, n_classes=2):
    recall = np.zeros(n_classes)
    peak = points[peak[2]] # indx is at 3th element of peak variable
    bbox_label, bbox_idx = find_bbox(peak, labels, calibs)
    if bbox_label:
        prm_target = generate_car_masks(points, bbox_label, calibs).reshape(-1)
    else:
        prm_target = np.zeros_like(preds)
    for cls in range(n_classes):
        preds_inds = preds==cls
        target_inds = prm_target==cls
        non_pred_inds = preds!=cls
        tp = (preds_inds[target_inds]).sum()
        fn = non_pred_inds[target_inds].sum()
        recall[cls] = float(tp / (tp+fn))

    return np.array(recall)

def iou_precision_crm(preds, targets,  n_classes=2):
    precision = np.zeros(n_classes)

    for cls in range(n_classes):
        preds_inds = preds==cls
        target_inds = targets==cls
        tp = (preds_inds[target_inds]).sum()
        fp = preds_inds.sum()-tp
        precision[cls] = float(tp / (fp+tp))

    return precision

def iou_recall_crm(preds, targets,  n_classes=2):
    recall = np.zeros(n_classes)

    for cls in range(n_classes):
        preds_inds = preds==cls
        target_inds = targets==cls
        non_pred_inds = preds!=cls
        tp = (preds_inds[target_inds]).sum()
        fn = non_pred_inds[target_inds].sum()
        recall[cls] = float(tp / (tp+fn))
        
    return recall

def find_bbox(point, labels, calibs):
    bboxes = get_bboxes(labels=labels, calibs=calibs)
    i=-1
    for label in labels:
        if label['type'] != b'DontCare':
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

def bbox_recall(labels, idx_list): # RECALL = TP / TP + FN
    tp = 0
    fn = 0
    dontcare_noncar = 0
    num_bbox = len(labels)
    for i,label in enumerate(labels):
        if label['type'] != b'DontCare':
            if label['type'] == b'Car':
                if idx_list[i] >0:
                    tp += 1
                else:
                    fn += 1
            else:
                dontcare_noncar += 1
        else:
            dontcare_noncar += 1

    # no bbox to detect
    if dontcare_noncar == num_bbox:
        return -2
    return tp/(tp+fn)

def bbox_precision(labels, idx_list, fp): #PRECISION = TP / TP + FP
    tp = 0
    dontcare_noncar = 0
    num_bbox = len(labels)
    for i,label in enumerate(labels):
        if label['type'] != b'DontCare':
            if label['type'] == b'Car':
                if idx_list[i] >0:
                    tp += 1
            else:
                dontcare_noncar += 1
                if idx_list[i]>0:
                    fp += 1
        else:
            dontcare_noncar += 1

    # no bbox to detect or no peak is detected
    if dontcare_noncar == num_bbox or tp+fp == 0:
        return -2
    return tp/(tp+fp)

def get_detected_bbox_num(labels, idx_list):
    tp = 0
    dontcare_noncar = 0
    num_bbox = len(labels)
    for i,label in enumerate(labels):
        if label['type'] == b'Car':
            if idx_list[i] >0:
                tp += 1
        else:
            dontcare_noncar += 1

    valid_bbox_num = num_bbox-dontcare_noncar
    detected_bbox_num = tp
    return detected_bbox_num, valid_bbox_num

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
        num_frags =max(3, len(peaks_idxs)//5)
    
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
    peak_centers.pop(0)
    peak_centers = np.array(peak_centers) # 3D info of peak centers
    
    return peak_centers, valid_peak_list, valid_indexes

def KNN(points, anchor, k=10):
    points = points.detach().cpu().clone()
    pcd=open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(points[:, 0:3])
    pcd_tree = open3d.geometry.KDTreeFlann(pcd)

    [k, idxs, _] = pcd_tree.search_knn_vector_3d(pcd.points[anchor], k)
    print("KNN k, idx ", k, idxs)
    return idxs