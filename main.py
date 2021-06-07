# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
"""from centerresponsegeneration.kitti_dataset import KITTI_Dataset
from centerresponsegeneration.models.BasicCNN_2d import BasicCNN
from centerresponsegeneration.models.SPCNN_Minkowski import Basic_SPCNN, UNet

import MinkowskiEngine as ME"""
from centerresponsegeneration.utils import *
import torch
import numpy as np
from centerresponsegeneration.config import *
from centerresponsegeneration.utils import *
from centerresponsegeneration.calibration import Calibration

# Press the green button in the gutter to run the script.

def train_basicCNN():
    print_hi('PyCharm')
    params = {'batch_size': 64,
              'shuffle': True}

    train_set = KITTI_Dataset( mode='training', use_val=True, num_points=num_points )
    training_generator = torch.utils.data.DataLoader(train_set, **params)

    test_set = KITTI_Dataset(mode='testing', use_val=True, num_points=num_points)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    model = BasicCNN( input_channels=3, input_points=111397)
    model.train()
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for epoch in range(250):
        running_loss = 0.0

        for i, data in enumerate(training_generator):
            print(data.shape)
            pc = data[:]['pc']
            crm_pc = data[:]['crm_pc']
            print(pc.shape)
            print(crm_pc.shape)
            
            # clear the gradient
            optimizer.zero_grad()

            #feed the input and acquire the output from network
            outputs = model(pc)

            #calculating the predicted and the expected loss
            crm_pc = np.transpose(crm_pc)

            loss = loss_func(outputs, crm_pc)

            #compute the gradient
            loss.backward()

            #update the parameters
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

def train_SPCNN(device):
    params = {'batch_size': 8,
              'shuffle': True}
    voxel_size = 0.2
    train_set = KITTI_Dataset( mode='training', use_val=True , num_points = num_points)
    training_generator = torch.utils.data.DataLoader(train_set, collate_fn=ME.utils.batch_sparse_collate,  **params)
                                                                                                 #make ME can consume batches
                                                    
    #print(train_set.__len__()) # 3712
    test_set = KITTI_Dataset(mode='testing', use_val=True)
    test_generator = torch.utils.data.DataLoader(test_set, collate_fn=ME.utils.batch_sparse_collate,
                                                 **params)
    
    outs = None
    pcs = None
    #model = Basic_SPCNN(in_points=num_points, in_feat=1, out_feat=1, D=1)
    model = UNet( in_nchannel=4, out_nchannel=1, D=3)
    model.train()
    #print(model)

    best_model = {}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    tot_iter = 0
    best_loss = float('inf')
    for epoch in range(1):
        accum_loss = 0.0
        accum_iter = 0
        for i, data in enumerate(training_generator):
            
            coords, pc, crm_pc = data
            
            """pc = data['pc']  # n x 3 (x,y,rel) -> features
            crm_pc = data['crm_pc'] # n x 1 ->labels
            rounded_pc = data['rounded_pc'] # n x 3-> coordinates in int
            """ 
            #print(pc.shape, rounded_pc.shape, crm_pc.shape)
            # clear the gradient
            optimizer.zero_grad()
            
            input = ME.SparseTensor(pc, coordinates=coords)
            crm_input = ME.SparseTensor(crm_pc, coordinates=coords)
            
            #feed the input and acquire the output from network
            outputs = model(input)
            
            
            #print("out:",outputs.shape)
            """
            print("pc:",pc[:,0:2].int().shape)
            print("crm:",crm_input.features.shape)
            """
            #calculating the predicted and the expected loss
           
            loss = loss_func(outputs.features, crm_input.features)

            #compute the gradient
            loss.backward()

            #update the parameters
            optimizer.step()

            # print statistics
            accum_loss += loss.item()
            
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 100 == 0 or tot_iter == 1:
                print(
                    f'Iter: {tot_iter}, Epoch: {epoch}, Loss: {accum_loss / accum_iter}'
                )
                # save model
                if best_loss > accum_loss :
                    print('best_loss :', accum_loss/accum_iter)
                    best_loss = accum_loss
                    best_model['epoch'] = epoch
                    best_model['model_state_dict'] = model.state_dict()
                    best_model['optimizer_state_dict'] = optimizer.state_dict()
                    best_model['loss'] = best_loss
                    
                    first_el_size = pc.shape[0] // params['batch_size'] + 1
                    pcs = pc[:first_el_size,:]
                    outs = outputs.features[:first_el_size,:].detach().numpy()
                    crm = crm_pc[:first_el_size,:]
                    
                    visualize_pointcloud( pcs, outs, idx=0)
                    visualize_pointcloud( pcs, crm, idx=0)
                    
                accum_loss, accum_iter = 0, 0
                
    torch.save(best_model, 'best_model.pt')
    visualize_pointcloud( pcs, outs, idx=0)
    #loading model
    """
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    """

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    root = "/Users/ezgicakir/Downloads/outputs_prm_from_train-3"
    filenames = os.listdir(root)

    for file in filenames:
        if 'prm' not in file and 'npy' in file:
            orig_pc_file = open (os.path.join(root_dir, data_train_path, file.replace('npy', 'bin')), 'rb')
            orig_pc = np.fromfile(orig_pc_file, dtype=np.float32).reshape(-1, 4)#[:,0:3]
            crm = np.load( os.path.join(root_dir, crm_train_path_pc, file)).astype(float)

            label_file = os.path.join(root_dir, labels_path, file.replace('npy', 'txt'))
            calibs = Calibration( os.path.join(root_dir, calib_train_path), file.replace('npy', 'txt'))
            lines = read_labels( label_file)
            #pc_filename = file.split('.')[0] + '_pc.' + file.split('.')[-1]
            #pc = np.load( os.path.join(root, pc_filename)).astype(float)

            out = np.load( os.path.join(root, file)).astype(float)
            pc = out[:,0:4]
            out = out[:,4:5]
            #prm_naive_file = os.path.join(root, file.replace('.npy', '_prm_naive.npy'))
            #if not os.path.exists(prm_naive_file):
            #    continue
            #prm_naive = np.load(prm_naive_file).astype(float)

            prm_file = os.path.join(root, file.replace('.npy', '_prm.npy'))
            if not os.path.exists(prm_file):
                continue
            prm = np.load(prm_file).astype(float)

            bboxes = get_bboxes(labels=lines, calibs=calibs)

            print(os.path.join(root, file))
            png_name = file.replace('npy', 'png')
            print(orig_pc)
            print(pc)
            print("crm limits: ", np.max(crm), np.min(crm))
            print("output limits: ", np.max(out), np.min(out))
            #print("prm limits: ", np.max(prm[:,0:3]), np.min(prm[:,0:3]))
            #print("prm limits: ", np.max(prm), np.min(prm))
            #print("prm_naive limits: ", np.max(prm_naive), np.min(prm_naive))
            print(np.argmax(prm, axis=0))
            #print(prm[17314,0], prm[4294,1],prm[9984,2],prm[ 18585,3])
            #prm = prm - np.min(prm[:,0:3])
            print("prm limits: ", np.max(prm[:,0:3]), np.min(prm[:,0:3]))
            visualize_pointcloud( orig_pc, crm, bboxes, idx=0)
            visualize_pointcloud( pc, out, bboxes , mult= 10, idx=0)
            #visualize_pointcloud( pc, prm, bboxes , idx=0, mult=1)
            for i in range(3):
                print(prm[:,i])
                visualize_pointcloud( pc, prm[:,i], bboxes , idx=i, mult=1000)
            #visualize_pointcloud( pc, prm_naive, bboxes , idx=0)
