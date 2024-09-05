# Code from https://github.com/jinfagang/3d_detection_kit/blob/master/gen_bev_image.py

import numpy as np
import cv2
from .utils import *
import random

class bev_generator(object):
    
    # generates BEV of point cloud by subsampling points with num_points
    def generate_BEV_matrix(pc, crm, num_points=100000):
        
        if len(crm.shape) == 1:
            crm = np.expand_dims(crm, axis=1)
        size = pc.shape[0]
        if size > num_points:
            l = list(range(0, size))
            random.shuffle(l)
            l = l[:num_points]
            pc = pc[l,:]
        
            crm = crm[l,:]
           
        pc = np.hstack([pc[:,0:1],pc[:,2:3],pc[:,3:4]])
     
        return pc, crm
        
    def subsample_pc_crm(pc, crm, num_points=100000):
        
        if len(crm.shape) == 1:
            crm = np.expand_dims(crm, axis=1)
            
        size = pc.shape[0]
        
        if size > num_points:
            l = list(range(0, size))
            random.shuffle(l)
            l = l[:num_points]
            pc = pc[l,:]
        
            crm = crm[l,:]
     
        return pc, crm

    # generates BEV of point cloud, values are set using Z values of point cloud
    def generate_BEV_map_Z( pc, lr_range=[-20, 20], bf_range=[-20, 20], res=0.05):
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        z_range = [ np.min(pc[:,2]), np.max(pc[:,2]) ]

        # filter point cloud
        f_filt = np.logical_and((x>bf_range[0]), (x<bf_range[1]))
        s_filt = np.logical_and((y>lr_range[0]), (y<lr_range[1]))
        filt = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filt).flatten()
        x = x[indices]
        y = y[indices]
        z = z[indices]

        # convert coordinates to
        x_img = (-y/res).astype(np.int32)
        y_img = (-x/res).astype(np.int32)
        # shifting image, make min pixel is 0,0
        x_img -= int(np.floor(lr_range[0]/res))
        y_img -= int(np.floor(bf_range[0]/res))

        # crop y to make it not bigger than 255
        pixel_values = np.clip(a=z, a_min=z_range[0], a_max=z_range[1])
        def scale_to_255(a, min, max, dtype=np.uint8):
            return (((a - min) / float(max - min)) * 255).astype(dtype)
        pixel_values = scale_to_255(pixel_values, min=z_range[0], max=z_range[1])

        # according to width and height generate image
        w = 1+int((lr_range[1] - lr_range[0])/res)
        h = 1+int((bf_range[1] - bf_range[0])/res)
        im = np.zeros([h, w], dtype=np.uint8)
        im[y_img, x_img] = pixel_values
        cropped_cloud = np.vstack([x, y, z]).transpose()
        return im, cropped_cloud

    # generate BEV using class response map, return BEV of CRM
    def generate_BEV_CRM( pc, crm, lr_range=[-20, 20], bf_range=[-20, 20], res=0.05):
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        # filter point cloud
        f_filt = np.logical_and((x>bf_range[0]), (x<bf_range[1]))
        s_filt = np.logical_and((y>lr_range[0]), (y<lr_range[1]))
        filt = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filt).flatten()

        x = x[indices]
        y = y[indices]
        z = z[indices]
        cropped_crm = crm[indices]

        # convert coordinates to
        x_img = (-y/res).astype(np.int32)
        y_img = (-x/res).astype(np.int32)
        # shifting image, make min pixel is 0,0
        x_img -= int(np.floor(lr_range[0]/res))
        y_img -= int(np.floor(bf_range[0]/res))

        # according to width and height generate image
        w = 1+int((lr_range[1] - lr_range[0])/res)
        h = 1+int((bf_range[1] - bf_range[0])/res)
        im = np.zeros([h, w], dtype=np.uint8)
        im[y_img, x_img] = cropped_crm * 255

        cropped_cloud = np.vstack([x, y, z]).transpose()
        return im, cropped_cloud, cropped_crm

    # generates BEV of point clouds using Z values and BEV of CRM using crm values
    def generate_BEV_CRM_and_Z( pc, crm, lr_range=[-20, 20], bf_range=[-20, 20], res=0.05):
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        z_range = [ np.min(pc[:,2]), np.max(pc[:,2]) ]
        # filter point cloud
        f_filt = np.logical_and((x>bf_range[0]), (x<bf_range[1]))
        s_filt = np.logical_and((y>lr_range[0]), (y<lr_range[1]))
        filt = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filt).flatten()
        x = x[indices]
        y = y[indices]
        z = z[indices]
        cropped_crm = crm[indices]

        # convert coordinates to
        x_img = (-y/res).astype(np.int32)
        y_img = (-x/res).astype(np.int32)
        # shifting image, make min pixel is 0,0
        x_img -= int(np.floor(lr_range[0]/res))
        y_img -= int(np.floor(bf_range[0]/res))

        pixel_values = np.clip(a=z, a_min=z_range[0], a_max=z_range[1])
        def scale_to_255(a, min, max, dtype=np.uint8):
            return (((a - min) / float(max - min)) * 255).astype(dtype)
        pixel_values = scale_to_255(pixel_values, min=z_range[0], max=z_range[1])

        # according to width and height generate image
        w = int((lr_range[1] - lr_range[0])/res)
        h = int((bf_range[1] - bf_range[0])/res)
        im_pc = np.zeros([h, w], dtype=np.uint8)
        im_pc[y_img, x_img] = pixel_values

        im_crm = np.zeros([h, w], dtype=np.uint8)
        im_crm[y_img, x_img] = cropped_crm[:,0] * 255

        cropped_cloud = np.vstack([x, y, z]).transpose()

        return im_pc, im_crm, cropped_cloud

    # crops point cloud
    def crop_pc( pc, lr_range=[-20, 20], bf_range=[-20, 20] ):
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        # filter point cloud
        f_filt = np.logical_and((x>bf_range[0]), (x<bf_range[1]))
        s_filt = np.logical_and((y>lr_range[0]), (y<lr_range[1]))
        filt = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filt).flatten()

        x = x[indices]
        y = y[indices]
        z = z[indices]
        cropped_cloud = np.vstack([x, y, z]).transpose()
        return cropped_cloud

    # crops point cloud and class response map
    def crop_pc_and_crm( pc, crm, lr_range=[-20, 20], bf_range=[-20, 20] ):
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        # filter point cloud
        f_filt = np.logical_and((x>bf_range[0]), (x<bf_range[1]))
        s_filt = np.logical_and((y>lr_range[0]), (y<lr_range[1]))
        filt = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filt).flatten()

        x = x[indices]
        y = y[indices]
        z = z[indices]
        cropped_cloud = np.vstack([x, y, z]).transpose()

        crm_x = crm[:,0]
        cropped_crm = crm_x[indices]
        """
        crm_y = crm[:,1]
        crm_z = crm[:,2]
        crm_y = crm_y[indices]
        crm_z = crm_z[indices]
        
        cropped_crm = np.vstack([crm_x, crm_y, crm_z]).transpose()
        """

        return cropped_cloud, cropped_crm

def main():
    path = '/Users/ezgicakir/Documents/Thesis/data/data_object_velodyne/training/velodyne/000000.bin'
    pc =  load_pc(path)
    crm = '/Users/ezgicakir/Documents/Thesis/data/crm/training/crm/000000.npy'
    crm_ =  np.load(crm).reshape(-1,1) # .astype(np.float16)

    impc, imcrm, cloud = bev_generator.generate_BEV_CRM_and_Z(pc, crm_)
    #im, cloud = bev_generator.generate_BEV_CRM(pc, crm_)
    print(imcrm.shape)
    cv2.imshow('pc', impc)
    cv2.waitKey(0)
    cv2.imshow('crm', imcrm)
    cv2.waitKey(0)

#main()