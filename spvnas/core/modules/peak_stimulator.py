import torch
import numpy as np
import torch.nn.functional as F
from torch import autograd
import core.utils as utils

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
    peak_response_maps_con = np.zeros((inputs.F.shape))
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
            # PRM is absolute of all channels
            prm = grad.detach().cpu().clone()
            prm = np.absolute( prm ) # shape: Nx4, 2D
            # 0 <= prm <= 1
            if normalize:
                #normalize gradient
                mins= np.amin(np.array(prm[prm>0.0]), axis=0)
                maxs = np.amax(np.array(prm), axis=0)
                prm = (prm-mins)/(maxs-mins)
                prm[prm==float('inf')] = 0.0
                prm[prm==float('-inf')] = 0.0

            prm = utils.segment_ground(points=inputs.F, preds=prm, distance_threshold=0.15)
            #prm = grad.sum(1).clone().clamp(min=0).detach().cpu()
            #prm = prm.sum(1) # sums columns
            #peak_response_maps.append( prm / prm.sum() )

            peak_response_maps.append(prm)
            peak_response_maps_con +=np.asarray( prm)
            #valid_peak_list contains indexes of valid peaks in center response map, shape: Mx3, e.g.[0,0,idx]
            valid_peak_list.append(peak_list[idx,:])
            i += 1

    #print("i ", i)
    if len(peak_response_maps) >0:
        # shape = len(valid_peak_list), 2
        valid_peak_list = torch.stack(valid_peak_list) # [1,1,N] -> dimension of each channels of it
        # peak responses of each valid peak list is concatanated vertically
        # shape = (len(valid_peak_list) * number_of_points_in_scene), channel_no_of_grad
        #peak_response_maps_con = torch.cat(peak_response_maps, 0)
        #peak_response_maps_con = sum(peak_response_maps)
        #print("# of peak responses and shape ", len(peak_response_maps), valid_peak_list.shape, peak_response_maps[0].shape)
        
    return valid_peak_list, peak_response_maps, peak_response_maps_con

