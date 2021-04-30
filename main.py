# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from classresponsegeneration.kitti_dataset import KITTI_Dataset
from classresponsegeneration.models.BasicCNN_2d import BasicCNN
from classresponsegeneration.models.SPCNN_Minkowski import Basic_SPCNN, UNet
import MinkowskiEngine as ME
import torch
import numpy as np
from classresponsegeneration.config import * 

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


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
    params = {'batch_size': 1,
              'shuffle': True}

    train_set = KITTI_Dataset( mode='training', use_val=True , num_points = num_points)
    training_generator = torch.utils.data.DataLoader(train_set, **params)
    print(train_set.__len__())
    test_set = KITTI_Dataset(mode='testing', use_val=True)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    #model = Basic_SPCNN(in_points=num_points, in_feat=1, out_feat=1, D=1)
    model = UNet( in_nchannel=2, out_nchannel=1, D=1)
    #print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    for epoch in range(5):
        running_loss = 0.0
        print("epoch :", epoch)
        for i, data in enumerate(train_set):
            
            pc = data['pc']  # n x 3 (x,y,rel)
            crm_pc = data['crm_pc'] # n x 1
            
            # clear the gradient
            optimizer.zero_grad()
            
            input = ME.SparseTensor(pc[:,0:2], coordinates=pc[:,0:2].int())
            crm_input = ME.SparseTensor(crm_pc, coordinates=pc[:,0:2].int())
            
            #feed the input and acquire the output from network
            outputs = model(input)
            """"
            print("out:",outputs.shape)
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
            running_loss += loss.item()
            if i % 100 == 0:
                print('epoch: %d, data: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss ))
                running_loss = 0.0
            


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #train_basicCNN()
    train_SPCNN(device)


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
