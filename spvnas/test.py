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
from torchpack.callbacks import (InferenceRunner, MaxSaver,
                                 Saver, SaverRestore, Callbacks)
from torchpack.callbacks.metrics import MeanSquaredError
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU, MSE

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
    #logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

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
    """elif 'spvcnn' in args.name.lower():
        model = spvcnn(args.name)
    elif 'mink' in args.name.lower():
        model = minkunet(args.name)
    else:
        raise NotImplementedError
    """
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
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow['test']


    trainer.before_train()
    trainer.before_epoch()

    # important
    model.eval()
    c=0
    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        #if c < 10:
        if True:
            _inputs = dict()
            for key, value in feed_dict.items():
                if not 'name' in key:
                    _inputs[key] = value.cuda()
    
            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.float().cuda(non_blocking=True)
            
            inputs.F.requires_grad = True
            outputs = model(inputs)
            
            # ===============================
            # PRM object to calculate gradient
            grad_output = outputs.new_empty(outputs.size())
            grad_output.zero_()
            
            # Set 1 to the max of predicted center points in gradient
            grad_output[torch.argmax(outputs, dim=0),0] = 1
            
            if inputs.F.grad is not None: 
                inputs.F.grad.zero_() # shape is outputs.shape[0]x4 , 2D 
                
            # Calculate peak response maps backwarding the output
            outputs.backward(grad_output, retain_graph=True)
            #print("sè¦ºfè¦ºrr:", grad_output[torch.argmax(outputs, dim=0)] ) 
    
            prm = inputs.F.grad.detach().sum(1).clone().clamp(min=0).cpu() # shape: outputs.shape[0], 1D
            
            # ===============================
            
            # ===============================
            # Naive way to calculate gradients
            """
            outputs.backward(torch.ones_like(outputs), retain_graph=True)
            prm = inputs.F.grad.detach().sum(1).clone().clamp(min=0).cpu() # shape: outputs.shape[0], 1D
            """
            # ================================
            #print("sè¦ºfè¦ºrr:", np.any(np.array ( np.logical_and(inputs.F.grad.cpu().detach()==1.0 , inputs.F.grad.cpu().detach())>0.0), axis=0) ) 
    
            loss = criterion(outputs, targets)
            print("loss: " , loss.item())
            #save the voxelized output and voxelized point cloud
            filename = feed_dict['file_name'][0] # file is list with size 1
            out = outputs.cpu()
            inp_pc = inputs.C.cpu()
            
            # outputs.shape[0]x5, first 4 column is pc, last 1 column is output
            concat_in_out = np.concatenate((inp_pc.detach(),out.detach()),axis=1) 
            
            #np.save( os.path.join(configs.outputs, filename.replace('bin', 'npy')), concat_in_out)
            np.save( os.path.join(configs.outputs, filename.replace('.bin', '_prm.npy')), prm)
    
            output_dict = {
                'outputs': outputs,
                'targets': targets
            }
            trainer.after_step(output_dict)
        c += 1
    trainer.after_epoch()


if __name__ == '__main__':
    main()