from centerresponsegeneration.utils import *
if __name__ == '__main__':

    root = "/Users/ezgicakir/Downloads/a"
    filenames = os.listdir(root)
    filenames = sorted(filenames, reverse=True)

    for file in filenames:
        sample = file[0:6]
        calibs = Calibration( '/Users/ezgicakir/Documents/Thesis/data/training/calib/', sample+'.txt')
        gt_label_file = f'/Users/ezgicakir/Documents/Thesis/data/training/label_2/{sample}.txt'
        gt_lines = read_labels( gt_label_file)
        gt_bboxes = get_bboxes(labels=gt_lines, calibs=calibs)

        pred_label_file = os.path.join(root, file.replace('npy', 'txt'))
        pred_lines = read_labels( pred_label_file)
        pred_bboxes = get_bboxes(labels=pred_lines, calibs=calibs)

        if 'crm' in file:
            out = np.load( os.path.join(root, file)).astype(float)
            pc = out[:,0:3]
            #Preds are in green, ground truth are in blue
            visualize_pointcloud( pc, out, pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes, mult=1, idx=0)

        else:
            orig_pc_file = open ('/Users/ezgicakir/Documents/Thesis/data/training/velodyne/'+sample+'.bin', 'rb')
            orig_pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
            out = np.load( os.path.join(root, file)).astype(float)
            #Preds are in green, ground truth are in blue
            visualize_pointcloud( orig_pc, out, pred_bboxes=pred_bboxes, gt_bboxes=gt_bboxes, mult=1, idx=0)
            #visualize_pointcloud_onlypreds( pc, out, pred_bboxes=pred_bboxes, mult=1, idx=0)
