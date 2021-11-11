import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import autograd
from core import utils

class PeakStimulation(autograd.Function):
    @staticmethod
    def forward(ctx, input, win_size, peak_filter, return_aggregation=False):
        ctx.num_flags = 4
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'

        # peak finding by getting peak in windows
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad1d(offset, float('-inf'))
        padded_maps = padding(input)

        batch_size, num_channels,n = padded_maps.size()
        element_map = torch.arange(0, n).float().view(1, 1, n)[:,:, offset:-offset]
        element_map = element_map.to(input.device)
        
        _, indices = F.max_pool1d(padded_maps, 
                                    kernel_size = win_size, stride=1, return_indices=True)

        peak_map = (indices == element_map)
        # peak filtering
        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            
            return input, peak_list, (input * peak_map).view( batch_size, num_channels, -1).sum(2) / \
                peak_map.view( batch_size, num_channels, -1).sum(2)
        else:
            return input, peak_list, None
            
    @staticmethod
    def backward(ctx, grad_crm, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        _, num_channels, n = input.size()
        grad_input = peak_map * grad_output.view( 1, num_channels,1)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=False, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, win_size, peak_filter, return_aggregation)
    
def prm_backpropagation(inputs, outputs, peak_list, peak_threshold=0.9, normalize=False):
    # PRM paper to calculate gradient
    grad_output = outputs.new_empty(outputs.size())
    grad_output.zero_()

    valid_peak_response_map = []
    peak_response_maps_con = torch.zeros(size=(inputs.shape[0],1))

    valid_peak_list = []
    avg_sum = 0.0
    for idx in range(peak_list.size(0)):
        peak_val = outputs[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2]]
        # if peak val * log(z_val) < peak_threshold
        # amplify Center Response Map values
        # x is forward direction in velo
        #if peak_val * np.log(np.absolute( points[peak_list[idx,2], 0] + 2)) > peak_threshold:
        if peak_val * torch.log(torch.abs( inputs[peak_list[idx,2], 0] + 2)).item() > peak_threshold:
            valid_peak_list.append(peak_list[idx])

    if len(valid_peak_list) == 0:
        return valid_peak_list, valid_peak_response_map, peak_response_maps_con, avg_sum

    for idx in range(len(valid_peak_list)):
        grad_output.zero_()
        # Set 1 to the max of predicted center points in gradient
        grad_output[valid_peak_list[idx][0], valid_peak_list[idx][1], valid_peak_list[idx][2]] = 1
        """
        # Set K nearest neighbors of peak as 1, backpropagate from a group of points
        k=1
        #knn_list = utils.KNN(points=inputs.F, anchor=peak_list[idx,2], k=k)
        knn_list = utils.KNN(points=inputs.F, anchor=valid_peak_list[idx][2], k=k)
        for n in knn_list:
            grad_output[valid_peak_list[idx][0], valid_peak_list[idx][1], n] = 1
            #grad_output[peak_list[idx, 0], peak_list[idx, 1], n] = 1
        """
        if inputs.grad is not None:
            inputs.grad.zero_() # shape is N x input_channel_num , 2D

        # Calculate peak response maps backwarding the output
        outputs.backward(grad_output, retain_graph=True)
        grad = inputs.grad # N x input_channel_num

        prm = torch.abs(grad)
        if normalize:
            prm = torch.max(prm, dim=1).values
            min = torch.min(prm[torch.gt(prm,0.0)])
            max = torch.max(prm)
            prm = (prm-min)/(max-min)
            prm = torch.clamp(prm, min=0.005, max=max)
        
        prm = prm.view(-1,1).cpu()
        valid_peak_response_map.append(prm)
        peak_response_maps_con +=prm

    return valid_peak_list, valid_peak_response_map, peak_response_maps_con

