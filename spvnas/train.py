import argparse
import sys
import os
import random
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
from torchpack.callbacks import (InferenceRunner, MaxSaver, MinSaver,
                                 Saver, SaverRestore, Callbacks, TFEventWriter)
from torchpack.callbacks.metrics import MeanSquaredError
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU, MSE, Shrinkage, MTE
from model_zoo import spvnas_cnn, spvcnn, spvnas_specialized, spvnas_cnn_specialized
from core import builder, utils
from core.modules.peak_stimulator import *
from core.calibration import Calibration

# torchpack dist-run -np 1 python train.py configs/kittti/default.yaml
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
    
    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2**32 - 1)
        
    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    #model = builder.make_model()
    if 'spvnas' == configs.model.name:
        model = spvnas_specialized(args.name, input_channels = configs.data.input_channels , pretrained=False)
        model.train()
        model = model.change_last_layer(configs.data.num_classes)
    elif 'spvcnn'== configs.model.name:
        if args.weights is None:
            model = spvcnn(args.name, input_channels = configs.data.input_channels, num_classes=configs.data.num_classes, pretrained=False)
        else:
            model = spvcnn(args.name, num_classes=configs.data.num_classes, pretrained=True, weights = args.weights)
        model.train()
    elif 'spvnas_cnn'== configs.model.name:
        if args.weights is None:
            model = spvnas_cnn(input_channels = configs.data.input_channels, num_classes=configs.data.num_classes, pretrained=False)
        else:
            if 'iou' in args.weights: # means we took weights after semantickitti training
                model = spvnas_cnn_specialized(args.name, num_classes=configs.data.num_classes, pretrained=True, weights = args.weights)
            else:
                model = spvnas_cnn(args.name, num_classes=configs.data.num_classes, pretrained=True, weights = args.weights)

        model.train()
    else:
        raise NotImplementedError

    print("Number of Params in Model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    #print("\nmodel: " ,model )
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)
    
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%H:%M")

    trainer = SemanticKITTITrainer(model=model,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               num_workers=configs.workers_per_gpu,
                               seed=configs.train.seed,
                               tfevent=configs.tfevent,
                               tfeventname=configs.tfeventname,
                                checkpoint=args.weights)

    t1_start = perf_counter()

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[InferenceRunner(
            dataflow[split],
            callbacks=[MSE(name=f'mse/{split}')])
                      for split in ['valid']
                  ] + [
                      MinSaver(scalar='mse/valid',name=dt_string, save_dir=configs.best_model ),
                      Saver(save_dir=configs.checkpoints),
                      TFEventWriter(save_dir=configs.tfevent+configs.tfeventname)
                  ]
    )
    """
    win_size = configs.prm.win_size # 5
    peak_threshold =  configs.prm.peak_threshold # 0.5
    tp, fn = 0,0
    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        feed_dict_cuda = dict()
        for key, value in feed_dict.items():
            if key in ['lidar', 'inverse_map']:
                feed_dict_cuda[key] = value.cuda()

        filename = feed_dict['file_name'][0] # file is list with size 1, e.g 000000.bin
        #print("\ncurrent file: ", filename)
        inputs = feed_dict_cuda['lidar']

        # outputs are got 1-by-1 in test phase
        inputs.F.requires_grad = True
        outputs = model(inputs) # voxelized output (N,1)

        # make outputs in shape [Batch_size, Channel_size, Data_size]
        if len(outputs.size()) == 2:
            outputs_bcn = outputs[None, : , :]
        outputs_bcn = outputs_bcn.permute(0,2,1)
        # peak backpropagation
        _ , peak_list, aggregation = peak_stimulation(input=outputs_bcn, win_size=win_size, peak_filter=model.module.mean_filter,
                                                      return_aggregation=True)
        #peak_list: [0,0,indx], peak_responses=list of peak responses, peak_response_maps_sum: sum of all peak_responses
        peak_list, peak_responses, peak_response_maps_sum = prm_backpropagation(inputs.F, outputs_bcn, peak_list,
                                                                                peak_threshold=peak_threshold, normalize=True)

        # convert the output Peak Response Maps to the original number of points
        #peak_response_maps_sum = peak_response_maps_sum[feed_dict_cuda['inverse_map'].F.long()]
        #np.save( os.path.join(configs.outputs, filename.replace('bin', 'npy')), peak_response_maps_sum.detach().numpy())

        # Calculate recall of peak detection
        from core.calibration import Calibration
        calib_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'calib', filename.replace('bin', 'txt'))
        calibs = Calibration( calib_file )
        #configs.data_path = ..samepath/velodyne, so remove /velodyne and add /label_2
        label_file = os.path.join (configs.dataset.root, '/'.join(configs.dataset.data_path.split('/')[:-1]) , 'label_2', filename.replace('bin', 'txt'))
        labels = utils.read_labels( label_file)
        bbox_found = [0] * len(labels)
        for p in peak_list:
            peak_ind = p.cpu()
            peak_coord = inputs.F[peak_ind[2]].cpu().detach()[0:3] # indx is at 3th element of peak variable
            # Find bbox that the peak belongs to
            _, bbox_idx = utils.find_bbox(peak_coord, labels, calibs)
            if bbox_idx>=0:
                bbox_found[bbox_idx] = 1

        tp_, fn_ = utils.tp_fn_peak(labels, bbox_found)
        tp += tp_
        fn += fn_

    print("Recall (TP/(TP+FN)) of peaks detected in boxes: ", tp / (tp+fn))
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
    """
if __name__ == '__main__':
    main()
