import os
import utils
from config import *
from calibration import *

class CRM_generator(object):

    def save_center_reponse_map( threshold = 2, _in3d = True):
        vehicles = [ b'Car', b'Van', b'Truck', b'Tram']
        N = len( os.listdir( os.path.join( root_dir, labels_path)) )

        for idx in range(N):
            labels = utils.read_labels( os.path.join(root_dir, labels_path), idx)
            points = utils.read_points( os.path.join(root_dir, data_train_path), idx)
            calibs = Calibration( os.path.join(root_dir, calib_train_path), idx)

            map = np.zeros((points.shape[0], 1), dtype=np.float32) #we will only update first column
            #while visualization, map will behave as RGB colors info
            labels_crm = []  # type, center_x, min_x, max_x, center_y, min_y, max_y, center_z, min_z, max_z

            for label in labels:
                if label['type'] in vehicles:
                    # x -> l, y -> w, z -> h
                    # Convert camera(image) coordinates to laser point cloud coordinates in meters
                    center = calibs.project_rect_to_velo(np.array([[label['x'], label['y'], label['z']]]))

                    # 3D object dim, heigth, width, length in meters
                    #h,w,l = label['h'], label['w'], label['l']

                    # Center point
                    x = center[0][0]
                    y = center[0][1]
                    z = center[0][2] #+ h/2 # normally z is the min value but here I set it to middle
                    center = [x,y,z]
                    # AABB Bounding box, nearest point and farthest point
                    #min_point_AABB = [x - l / 2, y - w / 2, z - h / 2 , ]
                    #max_point_AABB = [x + l / 2, y + w / 2, z + h / 2 , ]
                    """
                    # Filter laser points in this range
                    x_filt = np.logical_and(
                        (points[:,0]>min_point_AABB[0]), (points[:,0]<max_point_AABB[0]))
                    y_filt = np.logical_and(
                        (points[:,1]>min_point_AABB[1]), (points[:,1]<max_point_AABB[1]))
                    z_filt = np.logical_and(
                        (points[:,2]>min_point_AABB[2]), (points[:,2]<max_point_AABB[2]))
                    filt = np.logical_and(x_filt, y_filt)  # Must be both established
                    filt = np.logical_and(filt, z_filt)  # Must be both established
                    """
                    #labels = utils.set_normal_label(max_point_AABB, min_point_AABB, points, filt, _in3d)
                    labels =  utils.get_crm_of_object(points, center, threshold, _in3d = False)
              
                    map += labels
                    #map[:,0][filt] = labels
                    #vehicles to R, person to G, cyclist to B, others to R&G
                    color_ind = 0
                    if label['type'] in [ b'Car', b'Van', b'Truck', b'Tram']:
                        color_ind = 0
                    elif label['type'] in [ b'Pedestrian', b'Person_sitting']:
                        color_ind = 1
                    elif label['type'] in [ b'Cyclist']:
                        color_ind = 2
                    elif label['type'] in [ b'Misc']:
                        color_ind = 0
                    """
                    crm = [label['type'],
                           x, min_point_AABB[0], max_point_AABB[0],
                           y, min_point_AABB[1], max_point_AABB[1],
                           z, min_point_AABB[2], max_point_AABB[2] ]
                    for i in range(len(crm)):
                        crm[i] = str(crm[i])

                    labels_crm.append(crm)"""

            #self.visualize_pointcloud(points, map, color_ind)
            crm_path = os.path.join( root_dir, crm_train_path_pc)
            #label_path = os.path.join( root_dir, crm_train_path_labels)
            np.save( crm_path +'/%06d' % (idx), map)
            #with open(label_path+'/%06d.txt' % (idx), 'w') as x:
            #    x.write('\n'.join(' '.join(row) for row in labels_crm))

    def save_class_reponse_map_2D(class_response_path, N):

        for idx in range(N):
            labels = utils.read_labels( os.path.join(root_dir, labels_path), idx)
            points = utils.read_points( os.path.join(root_dir, data_train_path), idx)
            calibs = Calibration( os.path.join(root_dir, calib_train_path), idx)

            map = np.zeros((points.shape[0], 3)) #we will only update first column
            #while visualization, map will behave as RGB colors info

            for label in labels:
                if label['type'] != b'DontCare':
                    # x -> l, y -> w, z -> h
                    # Convert camera(image) coordinates to laser point cloud coordinates in meters
                    xyz = calibs.project_rect_to_velo(np.array([[label['x'], label['y'], label['z']]]))

                    # 3D object dim, heigth, width, length in meters
                    h,w,l = label['h'], label['w'], label['l']

                    # Center point
                    x = xyz[0][0]
                    y = xyz[0][1]
                    # AABB Bounding box, nearest point and farthest point
                    min_point_AABB = [x - l / 2, y - w / 2, ]
                    max_point_AABB = [x + l / 2, y + w / 2 , ]

                    # Filter laser points in this range
                    x_filt = np.logical_and(
                        (points[:,0]>min_point_AABB[0]), (points[:,0]<max_point_AABB[0]))
                    y_filt = np.logical_and(
                        (points[:,1]>min_point_AABB[1]), (points[:,1]<max_point_AABB[1]))

                    filt = np.logical_and(x_filt, y_filt)  # Must be both established


                    labels = utils.set_normal_label(max_point_AABB, min_point_AABB, points, filt, _2D=True)

                    if label['type'] in [ b'Car', b'Van', b'Truck', b'Tram']:
                        map[:, 0][filt] = labels
                    elif label['type'] in [ b'Pedestrian', b'Person_sitting']:
                        map[:, 1][filt] = labels
                    elif label['type'] in [ b'Cyclist']:
                        map[:, 2][filt] = labels
                    elif label['type'] in [ b'Misc']:
                        map[:, 0][filt] = labels
                        map[:, 1][filt] = labels

            #self.visualize_pointcloud(points, map)
            #np.save(class_response_path+'/%06d' % (idx),np.concatenate((points[:,0:3],map),axis=1))
            #np.save(class_response_path+'/%06d' % (idx),map)


def box_center_to_corner(data):
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

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    #corner_box = np.dot(
    #    rotation_matrix, bounding_box) + eight_points.transpose()
    corner_box = bounding_box + eight_points.transpose()
    return corner_box.transpose()

if __name__=='__main__':
    CRM_generator.save_center_reponse_map(_in3d = False)
    """
    orig_pc_file = open ('/Users/ezgicakir/Documents/Thesis/data/data_object_velodyne/training/velodyne/000001.bin', 'rb')
    pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
    crm = np.load( '/Users/ezgicakir/Documents/Thesis/data/dummy/000001.npy').astype(float)

    bboxes = []
    label_file = os.path.join('/Users/ezgicakir/Documents/Thesis/data/training/label_2/000001.txt')
    calibs = Calibration( '/Users/ezgicakir/Documents/Thesis/data/data_object_calib/training/calib', 1)

    lines = utils.read_labels( label_file)
    for data in lines:

        if data['type'] != b'DontCare':

            h = data['h'] # box height
            w = data['w'] # box width
            l = data['l']  # box length (in meters)
            x = data['x']
            y = data['y']
            z = data['z']
            ry = data['rotation_y']
            xyz = calibs.project_rect_to_velo(np.array([[x,y,z]]))

            t = (xyz[0][0], xyz[0][1], xyz[0][2])  # location (x,y,z) in camera coord.
            # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            bbox = [h,w,l,t,ry]
            bboxes.extend(box_center_to_corner(bbox))

    utils.visualize_pointcloud( pc, crm, bboxes, idx=0)
    """