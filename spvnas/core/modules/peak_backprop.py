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
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, offset = ctx.saved_variables
        #print("prehook nonzero gradout ", (grad_output!=0.0).sum(0))
        #print("prehook grad: ",torch.sum(grad_output) )
        return (input - offset) * grad_output, None
        #return grad_output, None

class PostHook(Function):

    @staticmethod
    def forward(ctx, input, norm_factor):
        ctx.save_for_backward(norm_factor)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        norm_factor, = ctx.saved_variables
        eps =1e-10
        zero_mask = norm_factor <= eps
        grad_input = grad_output / (torch.abs(norm_factor) + eps)
        grad_input[zero_mask.detach()] = 0
        #print("posthook zeroed num and min",zero_mask.sum(0), torch.min(norm_factor))
        #print("posthook grad", torch.sum(grad_output) )
        
        return None, grad_input
        #return None, grad_output

def pr_conv3d(self, inputs):

    offset = inputs.F.min().detach()    
    inputs.F = PreHook.apply(inputs.F, offset) 
    
    resp = spf.conv3d(inputs=inputs,
                      kernel=self.kernel,
                      kernel_size=self.kernel_size,
                      bias=self.bias,
                      stride=self.stride,
                      dilation=self.dilation,
                      transpose=self.t).detach()
    
    pos_weight = F.relu(self.kernel).detach()
    inputs.F = inputs.F-offset
    
    norm_factor = spf.conv3d(inputs=inputs,
                             kernel=pos_weight,
                             kernel_size=self.kernel_size,
                             bias=None,
                             stride=self.stride,
                             dilation=self.dilation,
                             transpose=self.t)
    resp.F = PostHook.apply(resp.F, norm_factor.F)
    return resp
