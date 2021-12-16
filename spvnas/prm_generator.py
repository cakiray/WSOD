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
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.modules.peak_stimulator import peak_stimulation, prm_backpropagation
from model_zoo import spvnas_cnn

def main() -> None:
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
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

    if not os.path.exists(configs.outputs):
        os.mkdir(configs.outputs)

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

    if 'spvnas_cnn' == configs.model.name:
        model = spvnas_cnn(input_channels = configs.data.input_channels, num_classes=configs.data.num_classes, weights=args.weights, pretrained=True)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)

    model.eval()
    datatype = 'test'
    model.module._patch()
    model.eval()

    w = args.weights.split('/')[-1][:-3]
    out_dir = os.path.join( configs.outputs, f'{w}_{configs.prm.peak_threshold}'  )

    win_size = configs.prm.win_size # 5
    peak_threshold =  configs.prm.peak_threshold # 0.5

    for feed_dict in tqdm(dataflow[datatype], desc='eval'):
        feed_dict_cuda = dict()
        for key, value in feed_dict.items():
            if key in ['lidar', 'inverse_map']:
                feed_dict_cuda[key] = value.cuda()

        filename = feed_dict['file_name'][0] # file is list with size 1, e.g 000000.bin
        inputs = feed_dict_cuda['lidar']

        # outputs are got 1-by-1 in test phase
        inputs.F.requires_grad = True
        outputs = model(inputs) # voxelized output (N,1)

        # make outputs in shape [Batch_size, Channel_size, Data_size]
        if len(outputs.size()) == 2:
            outputs_bcn = outputs[None, : , :]
        outputs_bcn = outputs_bcn.permute(0,2,1)
        # peak backpropagation
        peak_list, aggregation = peak_stimulation(input=outputs_bcn, win_size=win_size, peak_filter=model.module.mean_filter,
                                                      return_aggregation=True)
        #peak_list: [0,0,indx], peak_responses=list of peak responses, peak_response_maps_sum: sum of all peak_responses
        peak_list, peak_responses, prm_sum = prm_backpropagation(inputs.F, outputs_bcn, peak_list,
                                                                 peak_threshold=peak_threshold, normalize=True)

        # convert the output Peak Response Maps to the original number of points
        prm_sum = prm_sum[feed_dict_cuda['inverse_map'].F.long()]
        #np.save( os.path.join(configs.outputs, filename.replace('.bin', '_crm.npy')), np.concatenate( (inputs.F.cpu().detach().numpy(), outputs.cpu().detach().numpy()) , axis=1 ) )
        np.save( os.path.join(configs.outputs, filename.replace('.bin', '.npy')), prm_sum.detach().numpy())

if __name__ == '__main__':
    main()
