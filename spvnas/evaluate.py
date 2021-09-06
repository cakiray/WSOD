import argparse
import sys
import os
import numpy as np
from time import perf_counter
from tqdm import tqdm

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder, utils
from core.modules.peak_stimulator import *
from core.calibration import Calibration

from model_zoo import spvnas_specialized, spvcnn, spvnas_best, spvcnn_best, spvnas_cnn

def main() -> None:
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--weights', type=str, help='path to pretrained model weights')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataset = builder.make_dataset()
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    if 'spvnas' == configs.model.name:
        model = spvnas_best(net_id=args.name, weights=args.weights, configs=configs, input_channels=configs.data.input_channels)
    elif 'spvcnn' == configs.model.name:
        model = spvcnn_best(net_id=args.name, weights=args.weights, input_channels=configs.data.input_channels, num_classes=configs.data.num_classes)
    elif 'spvnas_cnn' == configs.model.name:
        model = spvnas_cnn(input_channels = configs.data.input_channels, num_classes=configs.data.num_classes, weights=args.weights, pretrained=True)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    model.eval()
    datatype = 'test'
    criterion = builder.make_criterion()
    model.module._patch()
    model.eval()

    t1_start = perf_counter()
    mbbox_recall,mbbox_precision =0.0, 0.0
    win_size = configs.prm.win_size # 5
    peak_threshold =  configs.prm.peak_threshold # 0.5
    iou_threshold = configs.prm.iout_threshold
    count, prec_count,recall_count = 0,0,0
    bbox_p, bbox_r = 0,0

    print(f"Win size: {win_size}, Peak_threshold: {peak_threshold}")
    n,r,p = 0,0,0
    for feed_dict in tqdm(dataflow[datatype], desc='eval'):
        if n < 10:
            n += 1
            _inputs = dict()
            for key, value in feed_dict.items():
                if key not in ['subsize', 'pc_file','file_name','calibs','labels','rot_mat', 'scale_factor']:
                    _inputs[key] = value.cuda()
            filename = feed_dict['file_name'][0] # file is list with size 1, e.g 000000.bin
            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.float().cuda(non_blocking=True)
            print("\ncurrent file: ", filename)
            # outputs are got 1-by-1 in test phase
            inputs.F.requires_grad = True
            outputs = model(inputs) # voxelized output (N,1)
            loss = criterion(outputs, targets)
            print("\n\nloss: ", loss)
            # make outputs in shape [Batch_size, Channel_size, Data_size]
            if len(outputs.size()) == 2:
                outputs_bcn = outputs[None, : , :]
            outputs_bcn = outputs_bcn.permute(0,2,1)
            # peak backpropagation
            peak_list, aggregation = peak_stimulation(outputs_bcn, return_aggregation=True, win_size=win_size,
                                                      peak_filter=model.module.mean_filter)
            #print( "peak_Sti peak len", len(peak_list),aggregation)

            #peak_list: [0,0,indx], peak_responses=list of peak responses, peak_response_maps_sum: sum of all peak_responses
            peak_list, peak_responses, peak_response_maps_sum = prm_backpropagation(inputs, outputs_bcn, peak_list,
                                                                                    peak_threshold=peak_threshold, normalize=True)
            #print("peak list after backprop ", len(peak_list))
            #save the subsampled output and subsampled point cloud
            out = outputs.cpu()
            inp_pc = inputs.F.cpu() # input point cloud
            # concat_in_out.shape[0]x5, first 4 column is pc, last 1 column is output
            concat_in_out = np.concatenate((inp_pc.detach(),out.detach()),axis=1)
            np.save( os.path.join(configs.outputs, filename.replace('bin', 'npy')), concat_in_out)
            if len(peak_list) >0:
                for i in range(len(peak_responses)):
                    prm = peak_responses[i]
                    np.save( os.path.join(configs.outputs, filename.replace('.bin', '_prm_%d.npy' % i)), prm)

            #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /calibs
            calib_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'calib', filename.replace('bin', 'txt'))
            calibs = Calibration( calib_file )
            #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /label_2
            label_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'label_2', filename.replace('bin', 'txt'))
            labels = utils.read_labels( label_file)
            bbox_found_indicator = [0] * len(labels) # 0 if no peak found in a bbox, 1 if a peak found in a bbox
            fp_bbox = 0
            print(f"Valid peak len:{len(peak_list)}, Number of cars: {utils.get_car_num(labels)}")
            #Calculate mprecision and mrecall of each peak_response individually
            for i in range(len(peak_list)):
                prm = np.asarray(peak_responses[i])
                peak_ind = peak_list[i].cpu() # [0,0,idx] idx in list inputs.F
                points = np.asarray(inputs.F[:,0:3].detach().cpu()) # 3D info of points in cloud
                peak = points[peak_ind[2]] # indx is at 3th element of peak variable

                # Find bbox that the peak belongs to
                bbox_label, bbox_idx = utils.find_bbox(peak, labels, calibs)
                if bbox_idx == -1: # if peak do not belong to any bbox, it is false positive
                    fp_bbox += 1
                    #print(f"FP (no gt bbox is related) CRM value: {outputs[peak_ind[2]]}, PRM value: {peak_responses[i][peak_ind[2]]}")

                #Mask of predicted PRM, points with positive value as 1, nonpositive as 0
                # If each channel of peaks are returned, shape=(N,channel_num)
                for col in range(prm.shape[1]):
                    mask_pred = utils.generate_prm_mask(prm[:,col])
                    # If peak belongs to a bbox
                    if bbox_idx > -1:
                        # generate mask for the bbox in interest
                        prm_target = utils.generate_car_masks(points, bbox_label, calibs).reshape(-1)
                        # if at least 1 channel of PRM has iou more that x%, it would be true positive
                        iou_bbox = utils.iou(mask_pred, prm_target, n_classes=2)
                        # if iou of peak's response and bbox is greater that 0.5, the peak is true positive
                        if iou_bbox[1] >= iou_threshold:
                            bbox_found_indicator[bbox_idx] = 1

                if bbox_idx >-1  and bbox_found_indicator[bbox_idx] == 1:
                    pass
                    #print(f"TP CRM value: {outputs[peak_ind[2]]}, PRM value: {peak_responses[i][peak_ind[2]]}")
                elif bbox_idx >- 1:
                    pass
                    #print(f"FP (under iou threshold) CRM value: {outputs[peak_ind[2]]}, PRM value: {peak_responses[i][peak_ind[2]]}")

            bbox_recall = utils.bbox_recall(labels, bbox_found_indicator)
            bbox_precision = utils.bbox_precision(labels, bbox_found_indicator, fp_bbox)

            if bbox_recall >= 0.0:
                mbbox_recall += bbox_recall
                bbox_r +=1

            if bbox_precision >= 0.0:
                mbbox_precision += bbox_precision
                bbox_p += 1

            count += len(peak_list)
        else:
            break
    mbbox_recall /= bbox_r
    mbbox_precision /= bbox_p

    print(f"\nMean Bbox Recall:{mbbox_recall}\nMean Bbox Precision:{mbbox_precision}\nTotal Number of PRMs: {count}")

    writer = SummaryWriter(configs.tfevent+configs.tfeventname)
    #writer.add_scalar(f"mBbox_Recall-ws_{win_size}-pt_{peak_threshold}-gs", mbbox_recall, 0)
    #writer.add_scalar(f"mBbox_Precision-ws_{win_size}-pt_{peak_threshold}-gs", mbbox_precision, 0)

    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
if __name__ == '__main__':
    main()
