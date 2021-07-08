import argparse
import sys
import os
import numpy as np

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

from model_zoo import spvnas_specialized, minkunet, spvcnn, spvnas_best

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


    if 'spvnas' in args.name.lower():
        model = spvnas_best(args.name, args.weights, configs)    
    elif 'spvcnn' in args.name.lower():
        model = spvcnn(args.name)
    elif 'mink' in args.name.lower():
        model = minkunet(args.name)
    else:
        raise NotImplementedError
    #model.change_last_layer(configs.data.num_classes)
    #model = builder.make_model()
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
    # important
    model.eval()

    miou = np.zeros(shape=(4,2)) #miou is calculated by binary class (being pos, being nonpos values on PRM)
    win_size = configs.prm.win_size # 5
    peak_threshold =  configs.prm.peak_threshold # 0.5
    count = 0 
    print(f"Win size: {win_size}, Peak_threshold: {peak_threshold}")
    c = 0
    for feed_dict in tqdm(dataflow[datatype], desc='eval'):
        if True:#c < 500:
            c += 1
            _inputs = dict()
            for key, value in feed_dict.items():
                if not 'name' in key:
                    _inputs[key] = value.cuda()

            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.float().cuda(non_blocking=True)
        
            # outputs are got 1-by-1
            inputs.F.requires_grad = True
        
            outputs = model(inputs) # voxelized output (N,1)
            loss = criterion(outputs, targets) 

            # make outputs in shape [Batch_size, Channel_size, Data_size]
            if len(outputs.size()) == 2:
                outputs_bcn = outputs[None, : , :]
            outputs_bcn = outputs_bcn.permute(0,2,1)
        
            # peak backpropagation
            peak_list, aggregation = peak_stimulation(outputs_bcn, return_aggregation=True, win_size=win_size, peak_filter=model.module.mean_filter)
            #print( "backprop calling ", len(peak_list),aggregation)
        
            #peak_list: [0,0,indx], peak_responses=list of peak responses, peak_response_maps_sum: sum of all peak_responses
            peak_list, peak_responses, peak_response_maps_sum = prm_backpropagation(inputs, outputs_bcn, peak_list,
                                                            peak_threshold=peak_threshold, normalize=False)
        
            #save the subsampled output and subsampled point cloud
            filename = feed_dict['file_name'][0] # file is list with size 1, e.g 000000.bin
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
            """ 
            
            #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /calibs
            calib_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'calib', filename.replace('bin', 'txt'))
            calibs = Calibration( calib_file )
            #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /label_2
            label_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'label_2', filename.replace('bin', 'txt'))
            labels = utils.read_labels( label_file)
            #Masked ground truth of instances, points in instances bbox as 1, remainings as 0
            mask_gt_prm = utils.generate_car_masks(np.asarray(inputs.F[:,0:3].detach().cpu()), labels,  calibs)

            # Calculate the mIoU of the sum of peak_responses
            if len(peak_list)>0:
                if peak_response_maps_sum.shape[1]>1:
                    ious = np.zeros(shape=(4,2))
                    for col in range(peak_response_maps_sum.shape[1]):
                        mask_pred = utils.generate_prm_mask(peak_response_maps_sum[:,col])
                        iou_col = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                        ious[col] = iou_col

                    if not np.isnan(np.sum(ious)):
                        miou += ious
                    count += 1

            # If there is no peak detected
            if len(peak_list) == 0:
                # If each channel of peaks are returned, shape=(N,4)
                if len(miou.shape)==2:
                    ious = np.zeros(shape=(4,2))
                    for col in range(miou.shape[0]):
                        mask_pred = np.zeros_like(mask_gt_prm)
                        iou_col = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                        ious[col] = iou_col
                    if not np.isnan(np.sum(ious)):
                        miou += ious
                else: # If sum of channels is detected
                    mask_pred = np.zeros_like(mask_gt_prm)
                    ious = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                    if not np.any(np.sum(ious)):
                        miou += ious
                count += 1

            """
            #Calculate mIoU of each peak_response individually
            for i in range(len(peak_list)):
                prm = np.asarray(peak_responses[i])
                #Mask of predicted PRM, points with positive value as 1, nonpositive as 0
                # If each channel of peaks are returned, shape=(N,4)
                if prm.shape[1] >1:
                    ious = np.zeros(shape=(4,2))
                    for col in range(prm.shape[1]):
                        mask_pred = utils.generate_prm_mask(prm[:,col])
                        iou_col = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                        ious[col] = iou_col
                    
                    if not np.isnan(np.sum(ious)):    
                        miou += ious

                else: # If sum of channels is detected
                    mask_pred = utils.generate_prm_mask(prm)
                    ious = utils.iou(mask_pred, mask_gt_prm, n_classes=2)
                    
                    if not np.isnan(np.sum(ious)):    
                        miou += ious
                count += 1
            """


            output_dict = {
                'outputs': outputs,
                'targets': targets
            }
            trainer.after_step(output_dict)
        else:
            break
    trainer.after_epoch()
    
    miou /= count
    print(f"mIoU:\n\t{miou},\nTotal Number of PRMs: {count}")

    writer = SummaryWriter(configs.tfevent+configs.tfeventname)
    for r,miou_col in enumerate(miou):
        #writer.add_scalar(f"prm-mIoU-pos-ws_{win_size}-pt_{peak_threshold}", miou_col[1], r)
        #writer.add_scalar(f"prm-mIoU-neg-ws_{win_size}-pt_{peak_threshold}", miou_col[0], r)
        pass

if __name__ == '__main__':
    main()
