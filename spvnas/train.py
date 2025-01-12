import argparse
import sys
import os
import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver, MinSaver,
                                 Saver, SaverRestore, Callbacks, TFEventWriter)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU, MSE, Shrinkage, MTE
from model_zoo import spvnas_cnn, spvnas_cnn_specialized
from core import builder

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
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    if 'spvnas_cnn'== configs.model.name:
        if args.weights is None:
            model = spvnas_cnn(input_channels = configs.data.input_channels, num_classes=configs.data.num_classes, pretrained=False)
        else:
            if 'iou' in args.weights: # means we took weights after semantickitti training
                model = spvnas_cnn_specialized(args.name, num_classes=configs.data.num_classes, pretrained=True, weights = args.weights)
            else:
                model = spvnas_cnn(input_channels = configs.data.input_channels, num_classes=configs.data.num_classes, pretrained=True, weights = args.weights)
        model.train()
    else:
        raise NotImplementedError

    print("Number of Params in Model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
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

if __name__ == '__main__':
    main()
