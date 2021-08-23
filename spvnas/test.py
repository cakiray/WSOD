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
from torchpack.callbacks import (InferenceRunner, MaxSaver,
                                 Saver, SaverRestore, Callbacks)
from torchpack.callbacks.metrics import MeanSquaredError
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder, utils
from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU, MSE
from core.modules.peak_stimulator import *
from core.calibration import Calibration 

from model_zoo import spvnas_specialized, minkunet, spvcnn, spvnas_best, myspvcnn

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


    if 'spvnas' in configs.model.name:
        model = spvnas_best(net_id=args.name, weights=args.weights, configs=configs, input_channels=configs.data.input_channels)
    elif 'spvcnn' in configs.model.name:
        model = myspvcnn(configs=configs, weights=args.weights, pretrained=True)
    elif 'mink' in configs.model.name:
        model = minkunet(args.name)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)
    model.eval()

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=configs.train.seed,
                                   out_save_dir=configs.outputs
                                   )
    callbacks=Callbacks([
        SaverRestore(),
        MSE()
    ])
    if args.weights is not None:
        trainer._load_state_dict(torch.load(args.weights))
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    
    datatype = 'test'
    trainer.dataflow = dataflow[datatype]

    trainer.before_train()
    trainer.before_epoch()

    model.module._patch()
    model.eval()
    
    t1_start = perf_counter()
    channel_num = configs.data.input_channels
    miou = np.zeros(shape=(channel_num,2)) #miou is calculated by binary class (being pos, being nonpos values on PRM)
    miou_crm = np.zeros(shape=(1,2)) #miou of crm is calculated by binary class (being pos, being nonpos values on PRM)
    mprecision = np.zeros(shape=(channel_num,2))
    mrecall = np.zeros(shape=(channel_num,2))
    mprecision_crm = np.zeros(shape=(1,2))
    mrecall_crm = np.zeros(shape=(1,2))
    mbbox_recall,mbbox_precision =0.0, 0.0
    win_size = configs.prm.win_size # 5
    peak_threshold =  configs.prm.peak_threshold # 0.5
    count, prec_count,recall_count = 0,0,0
    total_bbox_num , total_detected_bbox_num= 0, 0
    bbox_p, bbox_r = 0,0

    print(f"Win size: {win_size}, Peak_threshold: {peak_threshold}")
    n,r,p = 0,0,0 
    for feed_dict in tqdm(dataflow[datatype], desc='eval'):
        if n < 5:
            n += 1
            _inputs = dict()
            for key, value in feed_dict.items():
                if key not in ['subsize', 'pc_file','file_name','calibs','labels','rot_mat', 'scale_factor']:
                    _inputs[key] = value.cuda()

            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.float().cuda(non_blocking=True)
        
            # outputs are got 1-by-1 in test phase
            inputs.F.requires_grad = True
        
            outputs = model(inputs) # voxelized output (N,1)
            loss = criterion(outputs, targets)

            # make outputs in shape [Batch_size, Channel_size, Data_size]
            if len(outputs.size()) == 2:
                outputs_bcn = outputs[None, : , :]
            outputs_bcn = outputs_bcn.permute(0,2,1)
            points = np.asarray(inputs.F[:,0:3].detach().cpu()) # 3D info of points in cloud

            # peak backpropagation
            peak_list, aggregation = peak_stimulation(outputs_bcn, return_aggregation=True, win_size=win_size,
                                                      peak_filter=model.module.mean_filter)
            print( "peak_Sti peak len", len(peak_list),aggregation)
                        
            # peak_centers: 3D info of subsampled peaks, peak_list: subsampled peak_list
            peak_centers, peak_list = utils.FPS(peak_list, points, num_frags=-1)

            print("peak_list after FPS ", len(peak_list)) 
            #peak_list: [0,0,indx], peak_responses=list of peak responses, peak_response_maps_sum: sum of all peak_responses
            peak_list, peak_responses, peak_response_maps_sum = prm_backpropagation(inputs, outputs_bcn, peak_list,
                                                            peak_threshold=peak_threshold, normalize=True)
            print("peak list after backprop ", len(peak_list)) 
            #save the subsampled output and subsampled point cloud
            filename = feed_dict['file_name'][0] # file is list with size 1, e.g 000000.bin
             
            print("\ncurrent file: ", filename) 
            """
            out = outputs.cpu() 
            inp_pc = inputs.F.cpu() # input point cloud 
            # concat_in_out.shape[0]x5, first 4 column is pc, last 1 column is output
            concat_in_out = np.concatenate((inp_pc.detach(),out.detach()),axis=1) 
            np.save( os.path.join(configs.outputs, filename.replace('bin', 'npy')), concat_in_out)
            if len(peak_list) >0:    
                    
                for i in range(len(peak_responses)):
                    prm = peak_responses[i]
                    np.save( os.path.join(configs.outputs, filename.replace('.bin', '_prm_%d.npy' % i)), prm)
                            
                np.save(os.path.join(configs.outputs, filename.replace('.bin', '_prm.npy')), peak_response_maps_sum)
                np.save(os.path.join(configs.outputs, filename.replace('.bin', '_gt.npy')), targets.cpu())
            """
            #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /calibs
            calib_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'calib', filename.replace('bin', 'txt'))
            calibs = Calibration( calib_file )
            #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /label_2
            label_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'label_2', filename.replace('bin', 'txt'))
            labels = utils.read_labels( label_file)
            bbox_found_indicator = [0] * len(labels) # 0 if no peak found in a bbox, 1 if a peak found in a bbox
            fp_bbox = 0
            #Masked ground truth of instances, points in instances bbox as 1, remainings as 0
            mask_gt_prm = utils.generate_car_masks(np.asarray(inputs.F[:,0:3].detach().cpu()), labels,  calibs)

            # If there is no peak detected
            if len(peak_list) == 0:
                # If each channel of peaks are returned, shape=(N,channel_num)
                if len(miou.shape)==2:
                    ious = np.zeros(shape=(channel_num,2))
                    for col in range(miou.shape[0]):
                        mask_pred = np.zeros_like(mask_gt_prm)
                        iou_col = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                        ious[col] = iou_col
                    if not np.isnan(np.sum(ious)):
                        miou += ious
                else: # If sum of channels is detected
                    mask_pred = np.zeros_like(mask_gt_prm)
                    ious = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                    if not np.isnan(np.sum(ious)):
                        miou += ious
                count += len(peak_list)

            # Calculate the mIoU of the sum of peak_responses
            if len(peak_list)>0:
                if peak_response_maps_sum.shape[1]>1:
                    ious = np.zeros(shape=(channel_num,2))
                    for col in range(peak_response_maps_sum.shape[1]):
                        prm_c = np.asarray(peak_response_maps_sum[:,col])
                        mask_pred = utils.generate_prm_mask(prm_c)
                        iou_col = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                        ious[col] = iou_col

                    if not np.isnan(np.sum(ious)):
                        miou += ious
                    count += len(peak_list)

            #Calculate mprecision and mrecall of each peak_response individually
            #for i in range(len(peak_list)):
            for i in range(len(peak_list)):
                prm = np.asarray(peak_responses[i])
                peak_ind = peak_list[i].cpu() # [0,0,idx] idx in list inputs.F

                valid_peak = True
                for c in range(prm.shape[1]):
                    if prm[peak_ind[2]][c] < 1.0:
                        valid_peak = False
                valid_peak = True
                # If peak is not a valid peak, meaning has values lower than 1.0 at
                # each channel of prm, it is not considered in as detected
                # Because it is probably a false positive
                if valid_peak:
                    points = np.asarray(inputs.F[:,0:3].detach().cpu()) # 3D info of points in cloud
                    peak = points[peak_ind[2]] # indx is at 3th element of peak variable

                    # Find bbox that the peak belongs to
                    bbox_label, bbox_idx = utils.find_bbox(peak, labels, calibs)
                    if bbox_idx == -1: # if peak do not belong to any bbox, it is false positive
                        fp_bbox += 1
                        print(f"FP (no gt bbox is related) CRM value: {outputs[peak_ind[2]]}, PRM value: {peak_responses[i][peak_ind[2]]}")

                    #Mask of predicted PRM, points with positive value as 1, nonpositive as 0
                    # If each channel of peaks are returned, shape=(N,channel_num)
                    if prm.shape[1] >1:
                        prec = np.zeros(shape=(channel_num,2))
                        recall = np.zeros(shape=(channel_num,2))

                        for col in range(prm.shape[1]):
                            mask_pred = utils.generate_prm_mask(prm[:,col])
                            iou_prec= utils.iou_precision(peak_ind, points=points,
                                                          preds=mask_pred, labels=labels, calibs=calibs, n_classes=2)
                            iou_recall= utils.iou_recall(peak_ind, points=points,
                                                         preds=mask_pred, labels=labels, calibs=calibs, n_classes=2)
                            prec[col] = iou_prec
                            recall[col] = iou_recall

                            # If peak belongs to a bbox
                            if bbox_idx > -1:
                                # generate mask for the bbox in interest
                                prm_target = utils.generate_car_masks(points, bbox_label, calibs).reshape(-1)
                                # if at least 1 channel of PRM has iou more that 50%, it would be true positive
                                iou_bbox = utils.iou(mask_pred, prm_target, n_classes=2)
                                # if iou of peak's response and bbox is greater that 0.5, the peak is true positive
                                if iou_bbox[1] > 0.5:
                                    bbox_found_indicator[bbox_idx] = 1
                                else:
                                    print(  "IOU : ", iou_bbox[1])
                    if bbox_idx >-1  and bbox_found_indicator[bbox_idx] == 1:
                        print(f"TP CRM value: {outputs[peak_ind[2]]}, PRM value: {peak_responses[i][peak_ind[2]]}")
                    elif bbox_idx >- 1:
                        print(f"FP (under iou threshold) CRM value: {outputs[peak_ind[2]]}, PRM value: {peak_responses[i][peak_ind[2]]}")

                        if not np.isnan(np.sum(prec)):
                            mprecision += prec
                            prec_count += 1
                        if not np.isnan(np.sum(recall)):
                            mrecall += recall
                            recall_count += 1

            #Calculation of mean IoU on CRM
            crm = outputs.cpu()
            crm = crm.detach().numpy()
            crm = utils.segment_ground(inputs.F, crm, distance_threshold=0.15)
            mask_pred = utils.generate_prm_mask(crm)
            iou = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
            prec = utils.iou_precision_crm(crm, mask_gt_prm, n_classes=2 )
            recall = utils.iou_recall_crm(crm, mask_gt_prm, n_classes=2)
            bbox_recall = utils.bbox_recall(labels, bbox_found_indicator)
            bbox_precision = utils.bbox_precision(labels, bbox_found_indicator, fp_bbox)
            detected_bbox_num, valid_bbox_num = utils.get_detected_bbox_num(labels, bbox_found_indicator)
            total_bbox_num += valid_bbox_num
            total_detected_bbox_num += detected_bbox_num

            if bbox_recall >= 0.0:
                mbbox_recall += bbox_recall
                bbox_r +=1

            if bbox_precision >= 0.0:
                mbbox_precision += bbox_precision
                bbox_p += 1

            if not np.isnan(np.sum(prec)):
                mprecision_crm += prec
                p += 1
            if not np.isnan(np.sum(recall)):
                mrecall_crm += recall
                r += 1
            miou_crm += iou

            output_dict = {
                'outputs': outputs,
                'targets': targets
            }
            trainer.after_step(output_dict)
        else:
            break
    trainer.after_epoch()
    
    miou /= n
    miou_crm /= n
    mbbox_recall /= bbox_r
    mbbox_precision /= bbox_p
    mprecision_crm /= p
    mrecall_crm /= r
    mprecision /= prec_count
    mrecall /= recall_count
    bbox_detection_rate = total_detected_bbox_num / total_bbox_num
    print(f"Bbox Detection Rate:{bbox_detection_rate}")

    print(f"mIoU:\n{miou},\nmIoU CRM:{miou_crm}\nMean Bbox Recall:{mbbox_recall}\nMean Bbox Precision:{mbbox_precision}\nMean Precision:{mprecision}\nMean Recall:{mrecall}\nMean Precision CRM:{mprecision_crm}\nMean Recall CRM:{mrecall_crm}\nTotal Number of PRMs: {count}")

    writer = SummaryWriter(configs.tfevent+configs.tfeventname)
    for r,miou_col in enumerate(miou):
        #writer.add_scalar(f"prm-mIoU-pos-ws_{win_size}-pt_{peak_threshold}", miou_col[1], r)
        #writer.add_scalar(f"prm-mIoU-neg-ws_{win_size}-pt_{peak_threshold}", miou_col[0], r)

        #writer.add_scalar(f"prm-mPrecision-neg-ws_{win_size}-pt_{peak_threshold}-gs", mprecision[r][0], r)
        #writer.add_scalar(f"prm-mRecall-neg-ws_{win_size}-pt_{peak_threshold}-gs", mrecall[r][0], r)
        #writer.add_scalar(f"prm-mPrecision-pos-ws_{win_size}-pt_{peak_threshold}-gs", mprecision[r][1], r)
        #writer.add_scalar(f"prm-mRecall-pos-ws_{win_size}-pt_{peak_threshold}-gs", mrecall[r][1], r)
        pass
    
    #writer.add_scalar(f"crm-mPrecision-neg-ws_{win_size}-pt_{peak_threshold}-gs", mprecision_crm[0][0], 0)
    #writer.add_scalar(f"crm-mRecall-neg-ws_{win_size}-pt_{peak_threshold}-gs", mrecall_crm[0][0], 0)
    #writer.add_scalar(f"crm-mPrecision-pos-ws_{win_size}-pt_{peak_threshold}-gs", mprecision_crm[0][1], 0)
    #writer.add_scalar(f"crm-mRecall-pos-ws_{win_size}-pt_{peak_threshold}-gs", mrecall_crm[0][1], 0)
    #writer.add_scalar(f"mBbox_Recall-ws_{win_size}-pt_{peak_threshold}-gs", mbbox_recall, 0)
    #writer.add_scalar(f"mBbox_Precision-ws_{win_size}-pt_{peak_threshold}-gs", mbbox_precision, 0)

    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
if __name__ == '__main__':
    main()
