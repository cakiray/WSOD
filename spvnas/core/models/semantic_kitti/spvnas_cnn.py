import time
from collections import OrderedDict

import torch
import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor
from torchsparse.utils.kernel_region import *
from torchsparse.utils.helpers import *
from types import MethodType

from core.models.utils import *
from core.modules.layers import *
from core.modules.modules import *
from core.modules.networks import *
from core.modules.peak_backprop import *

__all__ = ['SPVNAS_CNN']

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transpose=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, midc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        midc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(midc),
            spnn.ReLU(True),
            spnn.Conv3d(midc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVNAS_CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        input_channels = kwargs.get('input_channels', 5)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [32,16,20,28,24,40,44,52,72,76,108,108,156,120,112,140,68,52,72,32,40,60,44,40,48]
        #cs = [32, 64, 128, 64, 32]
        cs = [int(cr * x) for x in cs]
        self.cs = cs
        self.pres = kwargs.get('pres', 0.05)
        self.vres = kwargs.get('vres', 0.05)

        self.stem = nn.Sequential(
            spnn.Conv3d(input_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], cs[3], ks=3, stride=1, dilation=1),
        )
        """
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[4], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[4], cs[5], cs[6], ks=3, stride=1, dilation=1),
        )


        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[6], cs[7], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[7], cs[8], cs[9], ks=3, stride=1, dilation=1),
        )
        """
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[9], cs[10], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[10], cs[11], cs[12], ks=3, stride=1, dilation=1),
        )
        """
        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[12], cs[13], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[13] + cs[9], cs[14], cs[15], ks=3, stride=1,
                              dilation=1),
            )]
        )

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[15], cs[16], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[16] + cs[6], cs[17], cs[18], ks=3, stride=1,
                              dilation=1),
            )
        ])
        """
        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[18], cs[19], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[19] + cs[3], cs[20], cs[21], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[21], cs[22], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[22] + cs[0], cs[23], cs[24], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[24],
                                                  kwargs['num_classes']))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[12]),
                nn.BatchNorm1d(cs[12]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[12], cs[18]),
                nn.BatchNorm1d(cs[18]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[18], cs[24]),
                nn.BatchNorm1d(cs[24]),
                nn.ReLU(True),
            )
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def shrink(self):
        cs= self.cs
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[10], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[10], cs[11], cs[12], ks=3, stride=1, dilation=1),
        )

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[12], cs[19], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[19] + cs[3], cs[20], cs[21], ks=3, stride=1,
                              dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[12]),
                nn.BatchNorm1d(cs[12]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[12], cs[21]),
                nn.BatchNorm1d(cs[21]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[12], cs[24]),
                nn.BatchNorm1d(cs[24]),
                nn.ReLU(True),
            )
        ])


    @staticmethod
    def median_filter(input):
        batch_size, num_channels, n = input.size()
        threshold = torch.median(input.view(batch_size, num_channels, n), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1)

    @staticmethod
    def mean_filter(input):
        # set non-zero to zero
        input = torch.max(input,torch.tensor([0.]).to(input.device))
        #get only positive values
        input_nonzero = input[:,:,torch.nonzero(input)[:,2]]

        batch_size, num_channels, n = input_nonzero.size()
        threshold = torch.mean(input_nonzero.view(batch_size, num_channels, n), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1)

    #modify conv3d layers so that they have prehook and posthook
    # to get PRM nicer
    def _patch(self):
        for name,module in self.named_modules():
            if isinstance(module, spnn.Conv3d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv3d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, spnn.Conv3d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def change_last_layer(self, num_classes):
        for name, param in self.named_parameters():
            param.requires_grad = True
        for name, module in self.named_modules():
            module.requires_grad_(True)
        # change last classifier layer's output channel and make it trainable, by default
        self.classifier = DynamicLinear(self.output_channels[-1], num_classes)
        self.classifier.set_output_channel(num_classes)
        return self

    def freezelayers(self, num_classes):
        # Freeze model weights
        for param in self.parameters():
            param.requires_grad = False
        # Change last layer to trainable
        self.classifier = nn.Sequential(nn.Linear(self.cs[8], num_classes))

    def forward_(self, x):

        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0) # 32 x 32
        z0 = voxel_to_point(x0, z, nearest=False) # 32
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0) # 32
        x1 = self.stage1(x1) # 64
        x2 = self.stage2(x1) # 128
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        out = self.classifier(z3.F)
        out = self.relu(out)
        return out



    def forward(self, x):

        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0) # 32 x 32
        z0 = voxel_to_point(x0, z, nearest=False) # 32
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0) # 32
        x1 = self.stage1(x1) # 64
        #x2 = self.stage2(x1) # 128
        #x3 = self.stage3(x2)
        x4 = self.stage4(x1)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)
        """
        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)
        """
        y3 = point_to_voxel(x4, z1)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[2](z1.F)
        out = self.classifier(z3.F)
        out = self.relu(out)

        return out