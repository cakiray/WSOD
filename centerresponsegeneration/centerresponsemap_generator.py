import os
import utils
from config import *
from calibration import *
import open3d

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

def fully_annot_crm():
    full_annot_path ='/data/Ezgi/fully_annot_crm'

    vehicles = [ b'Car']
    N = 1000#len( os.listdir( os.path.join( root_dir, labels_path)) )

    for idx in range(N):
        print(idx)
        labels = utils.read_labels( os.path.join(root_dir, labels_path), idx)
        points = utils.read_points( os.path.join(root_dir, data_train_path), idx)
        calibs = Calibration( os.path.join(root_dir, calib_train_path), idx)
        bboxes = utils.get_bboxes(labels=labels, calibs=calibs)

        #while visualization, map will behave as RGB colors info
        map = np.zeros((points.shape[0], 1), dtype=np.float32) #we will only update first column
        i=-1
        for label in labels:
            if label['type'] != b'DontCare':
                i += 1
            if label['type'] in vehicles:
                bbox = bboxes[i*8:(i+1)*8-1]
                obbox = open3d.open3d.geometry.OrientedBoundingBox()
                bboc_vec = open3d.utility.Vector3dVector(bbox)
                obbox = obbox.create_from_points(bboc_vec)
                points_vec = open3d.utility.Vector3dVector(points)
                b_points = obbox.get_point_indices_within_bounding_box(points_vec)

                map[b_points] = 1

        #utils.visualize_pointcloud_1bbox(points, map, bboxes=bboxes)

        np.save(full_annot_path+'/%06d' % (idx),map)



if __name__=='__main__':
    #save_center_reponse_map(_in3d = False)
    fully_annot_crm()

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
