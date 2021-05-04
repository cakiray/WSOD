from models.response_map_net import *
from dataloader import DataLoader
import torch

def main():

    dataloader = DataLoader()

    batch_size = 100

    train_x = torch.from_numpy(dataloader.dataset_train)
    train_y = torch.from_numpy(targets_train)
    test_x = torch.from_numpy(dataloader.dataset_test)
    test_y = torch.from_numpy(targets_test)

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x,train_y)
    test = torch.utils.data.TensorDataset(test_x,test_y)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)



if __name__=="__main__":
    main()