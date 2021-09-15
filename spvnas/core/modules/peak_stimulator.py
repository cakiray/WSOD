import torch
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
    #print("\nmax and min values in PRM: ", torch.max(outputs), torch.min(outputs))

    valid_peak_response_map = []
    peak_response_maps_con = np.zeros((inputs.F.shape))
    valid_peak_list = []
    avg_sum = 0.0
    for idx in range(peak_list.size(0)):
        peak_val = outputs[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2]]
        if peak_val > peak_threshold:
            valid_peak_list.append(peak_list[idx,:])

    if len(valid_peak_list) <= 0:
        return valid_peak_list, valid_peak_response_map, peak_response_maps_con, avg_sum
    # Further point sampling
    # peak_centers: 3D info of subsampled peaks, peak_list: subsampled peak_list
    points = np.asarray(inputs.F[:,0:3].detach().cpu())
    peak_centers, valid_peak_list, valid_indexes = utils.FPS(valid_peak_list, points, num_frags=-1)

    #for idx in range(peak_list.size(0)):
    for idx in range(len(valid_peak_list)):
        #peak_val = outputs[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2]]
        #if peak_val > peak_threshold:
        if True:
            grad_output.zero_()
            # Set 1 to the max of predicted center points in gradient
            #grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2]] = 1e10

            # Set K nearest neighbors of peak as 1, backpropagate from a group of points
            k=1
            #knn_list = utils.KNN(points=inputs.F, anchor=peak_list[idx,2], k=k)
            knn_list = utils.KNN(points=inputs.F, anchor=valid_peak_list[idx][2], k=k)
            for n in knn_list:
                grad_output[valid_peak_list[idx][0], valid_peak_list[idx][1], n] = 1
                #grad_output[peak_list[idx, 0], peak_list[idx, 1], n] = 1

            if inputs.F.grad is not None:
                inputs.F.grad.zero_() # shape is N x input_channel_num , 2D

            # Calculate peak response maps backwarding the output
            outputs.backward(grad_output, retain_graph=True)
        
            grad = inputs.F.grad # N x input_channel_num
            # PRM is absolute of all channels
            prm = grad.detach().cpu().clone()
            prm = np.asarray(np.absolute( prm )) # shape: N x input_channel_num, 2D
            #normalize gradient 0 <= prm <= 1
            if normalize:
                prm = utils.maxpool(prm) #channel no is 1 from now on
                #mins= np.amin(np.array(prm[prm>0.0]), axis=0)
                mins = np.asarray( [ np.amin(prm[prm[:,i]>0.0][:,i]) for i in range(prm.shape[1]) ] )
                maxs = np.amax(np.array(prm), axis=0)
                prm = (prm-mins)/(maxs-mins)
                #print("min max ", mins, maxs)
                prm[prm==float('inf')] = 0.0
                prm[prm==float('-inf')] = 0.0
                avg_sum += np.mean(prm[prm>0.0])
                prm[prm<0.001] = 0.0

                #prm = utils.assignAvgofNeighbors(points=inputs.F, prm=prm, k=10)
            #print("center and argmax center point ", points[np.argmax(prm)], points[valid_peak_list[idx][2]])
            #peak_response_maps.append(prm)
            valid_peak_response_map.append(prm)
            peak_response_maps_con +=prm
            #valid_peak_list contains indexes of valid peaks in center response map, shape: Mx3, e.g.[0,0,idx]
            #valid_peak_list.append(valid_peak_list[idx,:])

    #if len(peak_response_maps) >0:
    if False:
    #if len(valid_peak_response_map) >0:
        # shape = len(valid_peak_list), 2
        valid_peak_list = np.vstack(valid_peak_list) # [1,1,N] -> dimension of each channels of it
        #valid_peak_response_map = [peak_response_maps[i] for i in valid_indexes]
        """
        # Further point sampling 
        # peak_centers: 3D info of subsampled peaks, peak_list: subsampled peak_list
        points = np.asarray(inputs.F[:,0:3].detach().cpu())
        peak_centers, valid_peak_list, valid_indexes = utils.FPS(valid_peak_list, points, num_frags=-1)
        valid_peak_response_map = [peak_response_maps[i] for i in valid_indexes]
        """
        
        # peak responses of each valid peak list is concatanated vertically
        # shape = (len(valid_peak_list) * number_of_points_in_scene), channel_no_of_grad
        #peak_response_maps_con = torch.cat(peak_response_maps, 0)
        #peak_response_maps_con = sum(peak_response_maps)
        #print("# of peak responses and shape ", len(peak_response_maps), valid_peak_list.shape, peak_response_maps[0].shape)
    
    
    return valid_peak_list, valid_peak_response_map, peak_response_maps_con, avg_sum

