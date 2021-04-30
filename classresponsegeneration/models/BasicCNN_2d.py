import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self, input_channels, input_points):

        super(BasicCNN, self).__init__()
        self.input_points = input_points
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=5, padding=3)
        """
        self.linear1 = nn.Linear(776180,20)
        self.linear2 = nn.Linear(20,20)
        self.linear3 = nn.Linear(20,1)
        """

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        """
        x = x.view(-1, 776180)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        """
        return x