#The code is from https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/object3d_kitti.py

import numpy as np

def get_kitti_object_cloud_v2(self):

    save_object_cloud_path = r'D:\KITTI\Object\training\object_cloud'
    filecount = os.listdir(os.path.join(root_dir, data_train))

    for img_id in range(filecount):

        lidar_path =os.path.join(root_dir, data_train, '%06d.bin' % img_id)
        label_path =os.path.join(root_dir, label, '%06d.txt' % img_id)
        calib_path = os.path.join(root_dir, calib_train, '%06d.txt' % img_id )

        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
        labels = np.loadtxt(label_path,
                            dtype={'names': ('type', 'truncated', 'occuluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'h', 'w', 'l', 'x', 'y', 'z','rotation_y'),
                                   'formats': ('S14', 'float', 'float', 'float', 'float', 'float', 'float', 'float','float', 'float', 'float', 'float', 'float', 'float', 'float')})

        calibs = Calibration(calib_path)

        if labels.size == 1:
            labels = labels[np.newaxis]

        i = 0
        for label in labels:
            i += 1
            if label['type'] != b'DontCare':
                # Convert image coordinates to laser point cloud coordinates
                xyz = calibs.project_rect_to_velo(np.array([[label['x'], label['y'], label['z']]]))

                # Center point
                x = xyz[0][0]
                y = xyz[0][1]
                z = xyz[0][2]

                # AABB Bounding box, nearest point and farthest point
                min_point_AABB = [x - label['l'] / 2, y - label['w'] / 2, z, ]
                max_point_AABB = [x + label['l'] / 2, y + label['w'] / 2, z + label['h'], ]

                # Filter laser points in this range
                x_filt = np.logical_and(
                    (points[:,0]>min_point_AABB[0]), (points[:,0]<max_point_AABB[0]))
                y_filt = np.logical_and(
                    (points[:,1]>min_point_AABB[1]), (points[:,1]<max_point_AABB[1]))
                z_filt = np.logical_and(
                    (points[:,2]>min_point_AABB[2]), (points[:,2]<max_point_AABB[2]))
                filt = np.logical_and(x_filt, y_filt)  # Must be both established
                filt = np.logical_and(filt, z_filt)  # Must be both established

                object_cloud = points[filt, :]  # Filter

                # Records with only 1-3 points or records without points are not required, in fact, it can be more strict
                if object_cloud.shape[0] <= 3:
                    print('filter failed...', img_id, adjust_label, i)
                    continue

                #np.save(save_object_cloud_path+'\\%06d-%s-%d' % (img_id, adjust_label, i),object_cloud)

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                    % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                       self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str