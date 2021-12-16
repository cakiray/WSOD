from typing import Any, Callable, Dict
import numpy as np
import torch
from torch import nn
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['SemanticKITTITrainer']


class SemanticKITTITrainer(Trainer):
    def __init__(self, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler,
                 num_workers: int, seed: int,
                 tfevent:str=None, tfeventname:str=None, checkpoint=None) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.epoch_num = 1
        self.tfevent = tfevent
        self.tfeventname = tfeventname
        self.checkpoint = checkpoint

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num-1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
                self.seed + (self.epoch_num-1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:   
        feed_dict_cuda = dict()
        for key, value in feed_dict.items():
            if key in ['lidar', 'targets','inverse_map']:
                feed_dict_cuda[key] = value.cuda()

        inputs = feed_dict_cuda['lidar'] # voxelized input, .C is point cloud (N,4)
        targets = feed_dict['targets'].F.float().cuda(non_blocking=True)
        outputs = self.model(inputs) # voxelized output (N,1)
        
        if outputs.requires_grad:
            loss = self.criterion(outputs, targets)
            self.summary.add_scalar('loss', loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            pass
        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        if self.model.training:
            from torch.utils.tensorboard import SummaryWriter
            assert  self.tfevent!=None and self.tfeventname!=None

            writer = SummaryWriter(self.tfevent+self.tfeventname)
            for name, param in self.model.named_parameters():
                if 'bn' not in name:
                    writer.add_histogram(name, param.grad, self.epoch_num)
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['epoch'] = self.epoch_num
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def _before_train(self) -> None:
        if self.checkpoint is not None:
            checkpoint = torch.load(self.checkpoint)
            self.load_state_dict(checkpoint)
