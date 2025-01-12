import time
from collections import OrderedDict, deque
from types import MethodType

import torch
import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor
from torchsparse.utils.kernel_region import *
from torchsparse.utils.helpers import *

from core.models.utils import *

from core.modules.layers import *
from core.modules.modules import *
from core.modules.networks import *
from core.modules.peak_backprop import *


__all__ = ['SPVNAS']


class SPVNAS(RandomNet):
    base_channels = 32
    output_channels_lb = [base_channels, 16, 32, 64, 128, 128, 64, 48, 48]
    output_channels = [base_channels, 48, 96, 192, 384, 384, 192, 128, 128]
    max_macro_depth = 2
    max_micro_depth = 2
    num_down_stages = len(output_channels) // 2

    def __init__(self,input_channels, num_classes, macro_depth_constraint, **kwargs):
        super().__init__()
        self.pres = kwargs.get('pres', 0.05)
        self.vres = kwargs.get('vres', 0.05)
        self.cr_bounds = kwargs.get('cr_bounds', [0.125, 1.0])
        self.up_cr_bounds = [
            0.125, 1.0
        ] if 'up_cr_bounds' not in kwargs else kwargs['up_cr_bounds']
        self.trans_cr_bounds = [
            0.125, 1.0
        ] if 'trans_cr_bounds' not in kwargs else kwargs['trans_cr_bounds']

        if 'output_channels_ub' not in kwargs:
            self.output_channels_ub = self.output_channels
        else:
            self.output_channels_ub = kwargs['output_channels_ub']

        if 'output_channels_lb' in kwargs:
            self.output_channels_lb = kwargs['output_channels_lb']

        base_channels = self.base_channels
        output_channels = self.output_channels
        self.input_channel = input_channels
        self.stem = nn.Sequential(
            spnn.Conv3d(input_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels), spnn.ReLU(True),
            spnn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels), spnn.ReLU(True))

        num_down_stages = self.num_down_stages

        stages = []
        for i in range(1, num_down_stages + 1):
            stages.append(
                nn.Sequential(
                    OrderedDict([
                        ('transition',
                         DynamicConvolutionBlock(base_channels,
                                                 base_channels,
                                                 cr_bounds=self.trans_cr_bounds,
                                                 ks=2,
                                                 stride=2,
                                                 dilation=1)),
                        ('feature',
                         RandomDepth(
                             *[
                                 DynamicResidualBlock(base_channels,
                                                      output_channels[i],
                                                      cr_bounds=self.cr_bounds,
                                                      ks=3,
                                                      stride=1,
                                                      dilation=1),
                                 DynamicResidualBlock(output_channels[i],
                                                      output_channels[i],
                                                      cr_bounds=self.cr_bounds,
                                                      ks=3,
                                                      stride=1,
                                                      dilation=1)
                             ],
                             depth_min=macro_depth_constraint,
                         ))
                    ])))
            base_channels = output_channels[i]

        self.downsample = nn.ModuleList(stages)

        # take care of weight sharing after concat!
        upstages = []
        for i in range(1, num_down_stages + 1):
            new_base_channels = output_channels[num_down_stages + i]
            upstages.append(
                nn.Sequential(
                    OrderedDict([
                        ('transition',
                         DynamicDeconvolutionBlock(base_channels,
                                                   new_base_channels,
                                                   cr_bounds=self.up_cr_bounds,
                                                   ks=2,
                                                   stride=2)),
                        ('feature',
                         RandomDepth(
                             *[
                                 DynamicResidualBlock(
                                     output_channels[num_down_stages - i]
                                     + new_base_channels,
                                     new_base_channels,
                                     cr_bounds=self.up_cr_bounds,
                                     ks=3,
                                     stride=1,
                                     dilation=1),
                                 DynamicResidualBlock(
                                     new_base_channels,
                                     new_base_channels,
                                     cr_bounds=self.up_cr_bounds,
                                     ks=3,
                                     stride=1,
                                     dilation=1)
                             ],
                             depth_min=macro_depth_constraint,
                         ))
                    ])))
            base_channels = new_base_channels

        self.upsample = nn.ModuleList(upstages)

        self.point_transforms = nn.ModuleList([
            DynamicLinearBlock(output_channels[0],
                               output_channels[num_down_stages],
                               bias=True,
                               no_relu=False,
                               no_bn=False),
            DynamicLinearBlock(output_channels[num_down_stages],
                               output_channels[num_down_stages + 2],
                               bias=True,
                               no_relu=False,
                               no_bn=False),
            DynamicLinearBlock(output_channels[num_down_stages + 2],
                               output_channels[-1],
                               bias=True,
                               no_relu=False,
                               no_bn=False),
        ])

        self.classifier = DynamicLinear(output_channels[-1], num_classes)
        self.classifier.set_output_channel(num_classes)

        self.dropout = nn.Dropout(0.3, True)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.weight_initialization()
        
    def change_last_layer(self, num_classes):
        for name, param in self.named_parameters():
            param.requires_grad = True
        for name, module in self.named_modules():
            module.requires_grad_(True)
        
        # change last classifier layer's output channel and make it trainable, by default
        self.classifier = DynamicLinear(self.output_channels[-1], num_classes)
        self.classifier.set_output_channel(num_classes)
        return self
    
    def updatelayers_with_freeze(self, num_classes):
        # Freeze model weights except point_transforms' third layers block
        for name, param in self.named_parameters():
            #if 'point_transforms.2' not in name:
            param.requires_grad = False
        for name, module in self.named_modules():
            #if 'point_transforms.2' not in name:
            module.requires_grad_(False)
        # change last classifier layer's output channel and make it trainable, by default
        self.classifier = DynamicLinear(self.output_channels[-1], num_classes)
        self.classifier.set_output_channel(num_classes)
        return self

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

    # modify conv3d layers so that they have prehook and posthook
    # to get PRM nicer
    def _patch(self):
        for name,module in self.named_modules():
            if isinstance(module, spnn.Conv3d):
                if len(module.kernel.shape)>0:
                    #print("in c & out c ", module.in_channels, module.out_channels)
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

    def random_sample(self):
        sample = {}
        # sample layer configuration
        cur_outputs_channels = []
        for i in range(len(self.output_channels)):
            c = random.randint(self.output_channels_lb[i],
                               self.output_channels_ub[i])
            c = make_divisible(c)
            cur_outputs_channels.append(c)
        self.cur_outputs_channels = cur_outputs_channels
        sample['output_channels'] = cur_outputs_channels

        # fix point branch
        self.point_transforms[0].manual_select(
            self.cur_outputs_channels[self.num_down_stages])
        self.point_transforms[1].manual_select(
            self.cur_outputs_channels[self.num_down_stages + 2])
        self.point_transforms[2].manual_select(self.cur_outputs_channels[-1])

        # sample down blocks
        # all residual blocks, except the first one, must have inc = outc
        for i in range(len(self.downsample)):
            # sample output channels for transition block
            self.downsample[i].transition.random_sample()
            # sample depth
            cur_depth = self.downsample[i].feature.random_sample()

            # random sample each residual block
            for j in range(cur_depth):
                # random sample middile layers
                self.downsample[i].feature.layers[j].random_sample()
                # determine the output channel
                self.downsample[i].feature.layers[j].constrain_output_channel(
                    cur_outputs_channels[i + 1])

            for j in range(cur_depth, len(self.downsample[i].feature.layers)):
                self.downsample[i].feature.layers[j].clear_sample()

        # sample up blocks
        for i in range(len(self.upsample)):
            # sample output channels for transition block
            trans_output_channels = self.upsample[i].transition.random_sample()
            # sample depth
            cur_depth = self.upsample[i].feature.random_sample()
            # random sample each residual block
            for j in range(cur_depth):

                self.upsample[i].feature.layers[j].random_sample()
                self.upsample[i].feature.layers[j].constrain_output_channel(
                    cur_outputs_channels[len(self.downsample) + 1 + i])
                # special case: 1st layer for 1st residual block (because of concat)
                if j == 0:
                    cons = list(range(trans_output_channels)) + list(
                        range(
                            self.output_channels[len(self.downsample) + i + 1],
                            (self.output_channels[len(self.downsample) + i + 1]
                             + cur_outputs_channels[len(self.downsample) - 1
                                                    - i])))
                    self.upsample[i].feature.layers[j].net.layers[
                        0].constrain_in_channel(cons)
                    self.upsample[i].feature.layers[
                        j].downsample.constrain_in_channel(cons)

            for j in range(cur_depth, len(self.upsample[i].feature.layers)):
                self.upsample[i].feature.layers[j].clear_sample()

        for name, module in self.named_random_modules():
            try:
                cur_val = module.status()
                sample[name] = cur_val
            except BaseException:
                # random depth, ignored layer
                pass

        return sample

    def manual_select(self, sample):
        for name, module in self.named_random_modules():
            if sample[name] is not None:
                module.manual_select(sample[name])

        cur_outputs_channels = copy.deepcopy(sample['output_channels'])

        # fix point branch
        self.point_transforms[0].manual_select(
            cur_outputs_channels[self.num_down_stages])
        self.point_transforms[1].manual_select(
            cur_outputs_channels[self.num_down_stages + 2])
        self.point_transforms[2].manual_select(cur_outputs_channels[-1])

        for i in range(len(self.downsample)):
            for j in range(self.downsample[i].feature.depth):
                self.downsample[i].feature.layers[j].constrain_output_channel(
                    cur_outputs_channels[i + 1])

        for i in range(len(self.upsample)):
            trans_output_channels = self.upsample[i].transition.status()
            for j in range(self.upsample[i].feature.depth):
                self.upsample[i].feature.layers[j].constrain_output_channel(
                    cur_outputs_channels[len(self.downsample) + 1 + i])
                # special case: 1st layer for 1st residual block (because of concat)
                if j == 0:
                    cons = list(range(trans_output_channels)) + list(
                        range(
                            self.output_channels[len(self.downsample) + i + 1],
                            (self.output_channels[len(self.downsample) + i + 1]
                             + cur_outputs_channels[len(self.downsample) - 1
                                                    - i])))
                    self.upsample[i].feature.layers[j].net.layers[
                        0].constrain_in_channel(cons)
                    self.upsample[i].feature.layers[
                        j].downsample.constrain_in_channel(cons)

        self.cur_outputs_channels = cur_outputs_channels

    def determinize(self, local_rank=0):
        # Get the determinized SPVNAS network by running dummy inference.
        self.eval()
        sample_feat = torch.randn(1000, self.input_channel)
        sample_coord = torch.randn(1000, self.input_channel).random_(997)
        sample_coord[:, -1] = 0
        #x = SparseTensor(sample_feat,
        #                 sample_coord.int()).to('cuda:%d' % local_rank)
        if torch.cuda.is_available():
            x = SparseTensor(sample_feat,
                             sample_coord.int()).to('cuda:%d' % local_rank)
        else:
            x = SparseTensor(sample_feat,
                             sample_coord.int())
        with torch.no_grad():
            x = self.forward(x)

        model = copy.deepcopy(self)

        queue = deque([model])
        while queue:
            x = queue.popleft()
            for name, module in x._modules.items():
                while isinstance(module, RandomModule):
                    module = x._modules[name] = module.determinize()
                queue.append(module)

        return model


    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = point_to_voxel(x, z)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.downsample[0](x1)
        x2 = self.downsample[1](x1)
        x3 = self.downsample[2](x2)
        x4 = self.downsample[3](x3)

        # point transform 32 to 256
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.upsample[0].transition(y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.upsample[0].feature(y1)

        y2 = self.upsample[1].transition(y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.upsample[1].feature(y2)
        # point transform 256 to 128
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.upsample[2].transition(y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.upsample[2].feature(y3)

        y4 = self.upsample[3].transition(y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.upsample[3].feature(y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        self.classifier.set_in_channel(z3.F.shape[-1])
        out = self.classifier(z3.F)

        out = self.relu(out)
        
        return out
