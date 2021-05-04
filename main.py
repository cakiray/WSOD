# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from centerresponsegeneration.kitti_dataset import KITTI_Dataset
from centerresponsegeneration.models.BasicCNN_2d import BasicCNN
from centerresponsegeneration.models.SPCNN_Minkowski import Basic_SPCNN, UNet
from centerresponsegeneration.utils import * 
import MinkowskiEngine as ME
import torch
import numpy as np
from centerresponsegeneration.config import * 


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
    #train_basicCNN()
        
    train_SPCNN(device)


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
