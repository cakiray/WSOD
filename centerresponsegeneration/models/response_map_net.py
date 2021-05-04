import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv

# from https://github.com/traveller59/second.pytorch/blob/master/second/pytorch/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


class ResponseMapNet(nn.Module):
    def __init__(self, input_channels):
        super(ResponseMapNet, self).__init__()

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            spconv.SubMConv3d(32, 64, 3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.fc1 = nn.Linear(2**3*64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.25)
        self.relut = nn.LeakyReLu()

    def forward(self, x:torch.Tensor):
        x = self.net(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x