from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchsparse
from torchsparse.sparse_tensor import SparseTensor

import torchsparse.nn as spnn
import torchsparse.nn.functional as spf

class PreHook(Function):

    @staticmethod
    def forward(ctx, input, offset):
        ctx.save_for_backward(input, offset)
        """tensor = SparseTensor(input.F , input.C, input.s)
        tensor.coord_maps = input.coord_maps
        tensor.kernel_maps = input.kernel_maps
        return tensor"""
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, offset = ctx.saved_variables
        return (input - offset) * grad_output, None

class PostHook(Function):

    @staticmethod
    def forward(ctx, input, norm_factor):
        ctx.save_for_backward(norm_factor)
        """tensor = SparseTensor(input.F , input.C, input.s)
        tensor.coord_maps = input.coord_maps
        tensor.kernel_maps = input.kernel_maps
        return tensor"""
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        norm_factor, = ctx.saved_variables
        eps = 1e-10
        zero_mask = norm_factor < eps
        grad_input = grad_output / (torch.abs(norm_factor) + eps)
        grad_input[zero_mask.detach()] = 0
        return None, grad_input


def pr_conv3d(self, input):
    offset = input.F.min().detach()
    input.F = PreHook.apply(input.F, offset) 
    print("input ", input.F.shape)
    print("kernel ", self.kernel.shape)
    resp = spf.conv3d(inputs=input,
                      kernel=self.kernel,
                      kernel_size=self.kernel_size,
                      bias=self.bias,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t).detach()
    print("resp ", resp.F.shape)

    pos_weight = F.relu(self.kernel).detach()
    input.F = input.F-offset
    
    print("pos_weight ", pos_weight.shape)
    norm_factor = spf.conv3d(inputs=input, #- offset,
                             kernel=pos_weight,
                             kernel_size=self.kernel_size,
                             bias=None,
                             stride=self.stride,
                             dilation=self.dilation,
                             transpose=self.t)
    input.F = PostHook.apply(resp.F, norm_factor.F)
    print("norm_factor ", norm_factor.shape)
    print("input ", input.F.shape)

    return input
