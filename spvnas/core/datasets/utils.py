import os
import numpy as np

def generate_CRM_wfiles(radius, points, labels_path,  calibs_path, rot_mat, scale_factor ):
    vehicles = [ b'Car']

    labels = read_labels( labels_path )
    calibs = read_calibs( calibs_path)
    map = np.zeros((points.shape[0], 1), dtype=np.float32) #we will only update first column
    for label in labels:
        if label['type'] in vehicles:
            # x -> l, y -> w, z -> h
            # Convert camera(image) coordinates to laser point cloud coordinates in meters
            center = project_rect_to_velo(calibs, np.array([[label['x'], label['y'], label['z']]]))
            
            center = np.dot(center, rot_mat) * scale_factor
            # Center point
            x = center[0][0]
            y = center[0][1]
            z = center[0][2] #+ h/2 # normally z is the min value but here I set it to middle
            center = [x,y,z]

            crm =  get_distance(points, center, _in3d = False)
            crm =  standardize(crm, threshold=radius)
            crm = substact_1(crm)

            map += crm

    return map

def generate_CRM(radius, labels, points, calibs, rot_mat, scale_factor ):
    vehicles = [ b'Car']
    points[:, :3] = np.dot(points[:, :3], rot_mat) * scale_factor
    points = points.cpu()
    map = np.zeros((points.shape[0], 1), dtype=np.float32) #we will only update first column
    for label in labels:
        if label['type'] in vehicles:
            # x -> l, y -> w, z -> h
            # Convert camera(image) coordinates to laser point cloud coordinates in meters
            center = project_rect_to_velo(calibs, np.array([[label['x'], label['y'], label['z']]]))
            center = np.dot(center, rot_mat) * scale_factor

            # Center point
            x = center[0][0]
            y = center[0][1]
            z = center[0][2] #+ h/2 # normally z is the min value but here I set it to middle
            center = [x,y,z]

            crm =  get_distance(points, center, _in3d = False)
            crm =  standardize(crm, threshold=radius)
            crm = substact_1(crm)

            map += crm

    return map

def read_labels( label_path):
    label = np.loadtxt(label_path,
                       dtype={'names': ('type', 'truncated', 'occuluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'h', 'w', 'l', 'x', 'y', 'z','rotation_y'),
                              'formats': ('S14', 'float', 'float', 'float', 'float', 'float', 'float', 'float','float', 'float', 'float', 'float', 'float', 'float', 'float')})
    
    if label.size == 1:
        label = label[np.newaxis]

    return label

def read_points( lidar_path):
    file = open(lidar_path, 'rb')
    points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:,0:3]  # .astype(np.float16)
    return points

def normalize(arr):
    min = np.min(arr[arr>0.0])
    max = np.max(arr)
    arr = (arr-min)/(max-min)
    arr[arr<0.0] = 0.0
    return arr

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


def get_distance(points, center, _in3d = False):
    labels =  distance(points[:,0:3] , center, _in3d) # calculate distance of points to center point
    #labels[labels>=threshold] = 0 # discard the points that are away than threshold, meaning no longer belong to a object
    return labels

def standardize(data, threshold):
    labels_normalized = data / threshold # normalize labels
    return labels_normalized

def substact_1(data):
    data = 1- data # make center to be 1 and values decrease while getting away
    data[data<0] = 0
    return data.reshape(-1,1)


def project_ref_to_velo(pts_3d_ref, C2V):
    pts_3d_ref = cart2hom(pts_3d_ref) # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))

def project_rect_to_ref( pts_3d_rect, R0):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))

def project_rect_to_velo(calibs, pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
                     This part is what I want, I need to convert 3D coordinates to lidar coordinates
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, calibs['R0'])
    return project_ref_to_velo(pts_3d_ref, calibs['C2V'])

def cart2hom( pts_3d):
    ''' Input: nx3 points in Cartesian Cartesian is to add a column 1
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))

    return pts_3d_hom

def read_calibs(calibs_path):
    calibs = {}
    with open(calibs_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                calibs[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    # Projection matrix from rect camera coord to image2 coord
    P = calibs['P2']
    P = np.reshape(P, [3,4])
    calibs['P2'] = P
    # Rigid transform from Velodyne coord to reference camera coord
    V2C = calibs['Tr_velo_to_cam']
    V2C = np.reshape(V2C, [3,4])
    calibs['V2C'] = V2C

    C2V = inverse_rigid_trans(V2C)
    calibs['C2V'] = C2V
    # Rotation from reference camera coord to rect camera coord
    R0 = calibs['R0_rect']
    R0 = np.reshape(R0,[3,3])
    calibs['R0'] = R0

    return calibs


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr
