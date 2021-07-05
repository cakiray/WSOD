import torch
import numpy as np
import torch.nn.functional as F
from torch import autograd

import torchsparse
import torchsparse.nn as spnn


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
            return peak_list, (input * peak_map).view( batch_size, num_channels, -1).sum(2) / \
                peak_map.view( batch_size, num_channels, -1).sum(2)
        else:
            return peak_list, None
            
    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        _, num_channels, n = input.size()
        grad_input = peak_map * grad_output.view( 1, num_channels,1)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, win_size, peak_filter, return_aggregation)
    
def prm_backpropagation(inputs, outputs, peak_list, peak_threshold=0.08, normalize=False):
    # PRM paper to calculate gradient
    grad_output = outputs.new_empty(outputs.size())
    grad_output.zero_()
    prm = None
    #print("max and min values in PRM: ", torch.max(outputs), torch.min(outputs))
    
    valid_peak_list = []
    peak_response_maps = []
    i = 0
    for idx in range(peak_list.size(0)):
        peak_val = outputs[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2]]
        
        if peak_val > peak_threshold:
            #print("PEAK VALL ", peak_val)
            grad_output.zero_()
            # Set 1 to the max of predicted center points in gradient
            grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2]] = 1

            if inputs.F.grad is not None:
                inputs.F.grad.zero_() # shape is Nx4 , 2D

            # Calculate peak response maps backwarding the output
            outputs.backward(grad_output, retain_graph=True)
            #print("sè¦ºfè¦ºrr:", grad_output[torch.argmax(outputs, dim=0)] )

            grad = inputs.F.grad # Nx4
            #grad = torch.sum(grad[:,0:2],1)
            # 0 <= grad <= 1
            if normalize:
                grad = np.absolute(grad)
                #normalize gradient
                mins= np.amin(np.array(grad), axis=0)
                maxs = np.amax(np.array(grad), axis=0)
                grad = (grad-mins)/(maxs-mins)
                grad[grad==float('inf')] = 0

            # PRM is absolute and sum of all channels
            prm = grad.detach().cpu().clone()
            prm = np.absolute( prm ) # shape: Nx4, 2D
            #prm = grad.sum(1).clone().clamp(min=0).detach().cpu()
            #prm = prm.sum(1) # sums columns
            #peak_response_maps.append( prm / prm.sum() )
            peak_response_maps.append(prm)
            #valid_peak_list contains indexes of 2 dimensions of valid peaks in center response map
            valid_peak_list.append(peak_list[idx,:])
            i += 1

    #print("i ", i)
    if len(peak_response_maps) >0:
        # shape = len(valid_peak_list), 2
        valid_peak_list = torch.stack(valid_peak_list) # [1,1,N] -> dimension of each channels of it
        # peak responses of each valid peak list is concatanated vertically
        # shape = (len(valid_peak_list) * number_of_points_in_scene), channel_no_of_grad
        peak_response_maps_con = torch.cat(peak_response_maps, 0)
        #print("# of peak responses and shape ", len(peak_response_maps), valid_peak_list.shape, peak_response_maps[0].shape)
        
    return valid_peak_list, peak_response_maps
        
def peak_backpropagation_max(inputs, outputs, normalize=False):
    # PRM paper to calculate gradient
    grad_output = outputs.new_empty(outputs.size())
    grad_output.zero_()
    
    # Set 1 to the max of predicted center points in gradient
    grad_output[torch.argmax(outputs)] = 1

    if inputs.F.grad is not None:
        inputs.F.grad.zero_() # shape is Nx4 , 2D

    # Calculate peak response maps backwarding the output
    outputs.backward(grad_output, retain_graph=True)

    grad = inputs.F.grad.detach().cpu() # Nx4
    #grad = torch.sum(grad[:,0:2],1)
    if normalize:
        grad = np.absolute(grad)
        #normalize gradient
        mins= np.amin(np.array(grad), axis=0)
        maxs = np.amax(np.array(grad), axis=0)
        grad = (grad-mins)/(maxs-mins)
        grad[grad==float('inf')] = 0

    # 0 <= grad <= 1
    prm = grad # shape: Nx4, 2D
    # print(np.all( np.logical_and( np.array(grad)>=0.0, np.array(grad)<=1.0 ) )) #True,
            
    return prm
         
    
def median_filter(input):
    batch_size, num_channels, n = input.size()
    threshold = torch.median(input.view(batch_size, num_channels, n), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1)

def mean_filter(input, threshold = 0.0):
    batch_size, num_channels, n = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, n), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1)
    """
    #n,d  = inputs.shape
    n = len(input) 
    print("mean filter input shape ", input.shape)
    mean= torch.mean(input.view(-1,1), dim=0, keepdim=True)
    print("max ",torch.max(input))
    print("mean ", mean)
    return mean"""


def peak_stimulation_(input, filter='median', calc_threshold = True):
  
    # peak filtering
    # get mask to filter center response maps
    if filter == 'median':
        mask = input >= median_filter(input)
    elif filter == 'mean':
        mask = input >= mean_filter(input)
    else:
        assert NotImplementedError
        
    # peak_list constain the indexes of possible peaks on CRM, shape: [N,2]
    peak_list = torch.nonzero(mask)
    print("peak_list len ", peak_list.shape)
    
    if calc_threshold:
        temp = input[peak_list[:,0], peak_list[:,1]]
        peak_threshold = torch.mean( temp.view(-1,1), dim=0 )
        print(peak_threshold)
    
        return peak_list, peak_threshold
    else:
        return peak_list, -1
