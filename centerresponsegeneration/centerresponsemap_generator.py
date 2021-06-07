import os
import utils
from config import *
from calibration import *

class CRM_generator(object):

    # calculates center response maps and
    # save them as numpy array with .np extension
    def save_center_reponse_map( threshold = 3, _in3d = True):
        vehicles = [ b'Car']
        N = len( os.listdir( os.path.join( root_dir, labels_path)) )

        for idx in range(N):
            print(idx)
            labels = utils.read_labels( os.path.join(root_dir, labels_path), idx)
            points = utils.read_points( os.path.join(root_dir, data_train_path), idx)
            calibs = Calibration( os.path.join(root_dir, calib_train_path), idx)

            bboxes = utils.get_bboxes(labels=labels, calibs=calibs)

            #while visualization, map will behave as RGB colors info
            map = np.zeros((points.shape[0], 1), dtype=np.float32) #we will only update first column
            #utils.visualize_pointcloud_orig(points, boxes=bboxes)
            for label in labels:
                if label['type'] in vehicles:
                    # x -> l, y -> w, z -> h
                    # Convert camera(image) coordinates to laser point cloud coordinates in meters
                    center = calibs.project_rect_to_velo(np.array([[label['x'], label['y'], label['z']]]))

                    # Center point
                    x = center[0][0]
                    y = center[0][1]
                    z = center[0][2] #+ h/2 # normally z is the min value but here I set it to middle
                    center = [x,y,z]

                    crm =  utils.get_crm_of_object(points, center, _in3d = False)
                    #crm =  utils.standardize(crm, threshold=threshold)
                    #crm = utils.weight_data(crm)
                    #crm = utils.take_power(crm, threshold=1, power=3)
                    #crm = utils.substact_1(crm)
                    crm = utils.gaussian_distance(crm, c=2)
                    map += crm
                    #print(np.all( np.logical_and(crm>=0.0 , crm<= 1.0)))

            crm_path = os.path.join( root_dir, crm_train_path_pc)
            #np.save( crm_path +'/%06d' % (idx), map)
            #utils.visualize_pointcloud(points, map, mult=3, bboxes=bboxes, idx=0)
            utils.visualize_pointcloud_orig(points,bboxes)


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