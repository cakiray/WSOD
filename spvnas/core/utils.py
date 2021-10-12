import math
import os
import numpy as np
import struct
import open3d
import torch 
#from core.nms_gpu import nms_gpu

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

def bbox_recall(labels, idx_list): # RECALL = TP / TP + FN
    tp = 0
    fn = 0
    dontcare_noncar = 0
    num_bbox = len(labels)
    for i,label in enumerate(labels):
        if label['type'] == b'Car':
            if idx_list[i] >0:
                tp += 1
            else:
                fn += 1
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
        if label['type'] == b'Car':
            if idx_list[i] >0:
                tp += 1
        else:
            dontcare_noncar += 1
            if idx_list[i]>0:
                fp += 1

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
    #Eliminate points close each other more that 2.5 mt
    for i,center in enumerate(peak_centers):
        far = True
        for new_center in new_peak_centers:
            dist = L2dist_2d(center, new_center)
            if dist < 2.5:
                far=False
                break
                
        if True:#far:
            new_peak_centers.append(center)
            new_valid_peak_list.append(valid_peak_list[i])
            new_valid_indexes.append(valid_indexes[i])
    
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
        mask = response.flatten()>0.0
        pc_ = open3d.utility.Vector3dVector(points[mask][:,0:3])
        bbox = open3d.geometry.AxisAlignedBoundingBox()
        bbox = bbox.create_from_points(pc_)

        corners_o3d = bbox.get_box_points() #open3d.utility.Vector3dVector
        np_corners = np.asarray(corners_o3d) #Numpy array, 8x3
        #corners from velodyne to rect
        np_corners = calibs.project_velo_to_rect(np_corners) # 8x3
        #2D bounding box's corners location on image
        corners_img = calibs.corners3d_to_img_boxes(np.asarray([np_corners])) # 1x4

        #get center of bbox and convert from velo to rect
        np_center = bbox.get_center().reshape(1,3) #numpy, 1x3
        np_center = calibs.project_velo_to_rect(np_center) # x,y,z in velo -> x,z,y in rect

        # h->z, w->x, l->y
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        dimensions = max_bound-min_bound #length of each side of bounding box, eg. x,y,z:w,l,h
        h, w, l = dimensions[2], dimensions[0], dimensions[1]

        x, y, z = np_center[0,0], np_center[0,1]+h/2, np_center[0,2]
        ry = 0 # rotation along y axis is set to 0 for now
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry
        #score = crm[peak_list[i][2]].item() * points[peak_list[i][2],0] / 5 # confidence score
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

def non_maximum_supression(points, crm, peak_list, peak_responses,calibs):
    bboxs_raw = get_kitti_format(points, crm, peak_list, peak_responses, calibs)
    dets = np.zeros(shape=(len(peak_list), 5))

    for i,bbox in enumerate(bboxs_raw):
        x1 = bbox[3]#left (smaller than right)
        y1 = bbox[6]#bottom (bigger than top)
        x2 = bbox[5]#right (bigger than left)
        y2 = bbox[4]#top (smaller than bottom)
        score = bbox[-1]
        #bottom,top naming is different in nms and kitti format.
        dets[i] = np.asarray([[x1,y1,x2,y2,score]])

    #kept_idxs = nms_gpu(dets, nms_overlap_thresh=0.7, device_id=0) #gpu gave error
    kept_idxs = nms_cpu(dets, thresh=0.9)

    return kept_idxs

#cpu base NMS algorithm
def nms_cpu(dets, thresh, eps=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
            i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            # ovr = inter / areas[j]
            if ovr >= thresh:
                suppressed[j] = 1
    return keep

def save_in_kitti_format(file_id, kitti_output, points, crm, peak_list, peak_responses, calibs, labels):

    kitti_output_file = os.path.join(kitti_output, f'{file_id}.txt')
    kitti_format_list = get_kitti_format(points, crm, peak_list, peak_responses, calibs)
    kept_idxs = non_maximum_supression(points, crm, peak_list, peak_responses,calibs)
    #print(f"\n len kept: {len(kept_idxs)}, len original: {len(peak_list)}")

    with open(kitti_output_file, 'w') as f:
        for idx in kept_idxs:
            kitti_format = kitti_format_list[idx]
            print('%s 0.0 3 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  kitti_format, file=f)

            """img_boxes_w = corners_img[:, 2] - corners_img[:, 0]
           img_boxes_h = corners_img[:, 3] - corners_img[:, 1]
           """


def assignAvgofNeighbors(points, prm, k=10):
    for i in prm:
        knn_list = KNN(points=points, anchor=i, k=k)
        positive = True
        for n in knn_list:
            if prm[n,0] <0:
                positive=False
        if positive:
            prm[n] = np.average(np.array( [prm[t] for t in knn_list] ))

    return prm

# maxpool of Peak Response Map columns
def maxpool(prm):
    new_prm = np.amax(prm,axis=1).reshape(-1,1) #maxs columns-wise
    return new_prm
