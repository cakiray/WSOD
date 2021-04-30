import os
import utils
from config import *
from calibration import *

class CRM_generator(object):

    def save_class_reponse_map_3D(class_response_path, N):

        for idx in range(N):
            labels = utils.read_labels( os.path.join(root_dir, labels_path), idx)
            points = utils.read_points( os.path.join(root_dir, data_train_path), idx)
            calibs = Calibration( os.path.join(root_dir, calib_train_path), idx)

            map = np.zeros((points.shape[0], 1), dtype=np.float32) #we will only update first column
            #while visualization, map will behave as RGB colors info
            labels_crm = []  # type, center_x, min_x, max_x, center_y, min_y, max_y, center_z, min_z, max_z

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
                    z = xyz[0][2] + h/2 # normally z is the min value but here I set it to middle

                    # AABB Bounding box, nearest point and farthest point
                    min_point_AABB = [x - l / 2, y - w / 2, z - h / 2 , ]
                    max_point_AABB = [x + l / 2, y + w / 2, z + h / 2 , ]

                    # Filter laser points in this range
                    x_filt = np.logical_and(
                        (points[:,0]>min_point_AABB[0]), (points[:,0]<max_point_AABB[0]))
                    y_filt = np.logical_and(
                        (points[:,1]>min_point_AABB[1]), (points[:,1]<max_point_AABB[1]))
                    z_filt = np.logical_and(
                        (points[:,2]>min_point_AABB[2]), (points[:,2]<max_point_AABB[2]))
                    filt = np.logical_and(x_filt, y_filt)  # Must be both established
                    filt = np.logical_and(filt, z_filt)  # Must be both established

                    labels = utils.set_normal_label(max_point_AABB, min_point_AABB, points, filt)
                    map[:,0][filt] = labels
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

                    crm = [label['type'],
                           x, min_point_AABB[0], max_point_AABB[0],
                           y, min_point_AABB[1], max_point_AABB[1],
                           z, min_point_AABB[2], max_point_AABB[2] ]
                    for i in range(len(crm)):
                        crm[i] = str(crm[i])

                    labels_crm.append(crm)

            #self.visualize_pointcloud(points, map, color_ind)
            np.save(class_response_path+'/crm/%06d' % (idx), map)
            with open(class_response_path+'/labels/%06d.txt' % (idx), 'w') as x:
                x.write('\n'.join(' '.join(row) for row in labels_crm))

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
