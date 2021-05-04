from classresponsegeneration.config import *

from calibration import *
from classresponsemap_generator import CRM_generator

class DataLoader(object):
    def __init__(self, useval=False):
        self.N = len(os.listdir( os.path.join(root_dir, labels_path) ))
        """    
        if not useval:
            self.data_trn_count = len(os.listdir( os.path.join(root_dir, data_train_path) ))
            self.data_tst_count = len(os.listdir( os.path.join(root_dir, data_test_path) ))
            # Read train and test set as numpy array
            self.dataset_train = [] #np.zeros(shape=(self.data_trn_count,))
            self.dataset_test = [] #np.zeros(shape=(self.data_tst_count,))

            for idx in range(self.data_trn_count):

                data_train = self.read_bin_velodyne(os.path.join(root_dir, data_train_path) , idx)
                self.dataset_train.append(data_train)
                    #self.dataset_train[idx] = data_train

            for idx in range(self.data_tst_count):

                data_test = self.read_bin_velodyne(os.path.join(root_dir, data_test_path) , idx)
                self.dataset_test.append(data_test)
                    #self.dataset_test[idx] = data_test

        else:
            train_idxs = open( os.path.join(root_dir, "train.txt") ).readlines()
            val_idxs = open( os.path.join(root_dir, "val.txt") ).readlines()

            self.dataset_train = []
            self.dataset_test = []

            # Read train and test set as numpy array
            for (i,idx) in enumerate(train_idxs):
                data_train = self.read_bin_velodyne(os.path.join(root_dir, data_train_path) ,idx )
                self.dataset_train.append(data_train)

            for (i,idx) in enumerate(val_idxs):
                data_test = self.read_bin_velodyne(os.path.join(root_dir, data_train_path), idx )
                self.dataset_test.append(data_test)

        self.dataset_train = np.asarray(self.dataset_train, dtype=object)
        self.dataset_test = np.asarray(self.dataset_test, dtype=object)

        """"""print(np.asarray(self.dataset_train, dtype=object).shape) (31,)
        print(np.asarray(self.dataset_train)[0].shape) (115384, 3)
        print(np.asarray(self.dataset_test, dtype=object).shape) (31,)
        print(np.asarray(self.dataset_test)[0].shape)  (115384, 3)
        """"""

        """


def main():
    """
    filename=os.listdir(os.path.join(root_dir, data_train) )
    filename.sort()
    file_number=len(filename)

    pcd=open3d.open3d.geometry.PointCloud()

    for i in range(3):
        path=os.path.join(root_dir, data_train, filename[i])
        print(path)
        example=read_bin_velodyne(path)
        # From numpy to Open3D
        pcd.points= open3d.open3d.utility.Vector3dVector(example)
        open3d.open3d.visualization.draw_geometries([pcd])
    """

    CRM_generator.save_class_reponse_map_3D( os.path.join(root_dir, crm_train_path), DataLoader.N)
    #dataloader.save_class_reponse_map_2D( os.path.join(root_dir, crm_train_path))

if __name__=="__main__":
    main()