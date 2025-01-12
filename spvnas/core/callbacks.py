from typing import Any, Dict
import numpy as np
import torch

from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback


__all__ = [
    'MeanIoU',
    'MSE',
    'Shrinkage',
    'MTE'
]
class Shrinkage(Callback):
    def __init__(self,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'shrinkage',
                 a: float= 10.0,
                 c: float = 0.2) -> None:
        self.name = name
        self.a = a
        self.c = c
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor

    def _before_epoch(self):
        self.size = 0
        self.errors = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        l = torch.abs(outputs - targets)
        l2 = l ** 2
        deniminator = 1 + torch.exp(self.a* (self.c-l))
        error = torch.mean(l2/deniminator)
        #print(error.size(), error.item())
        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)
        #self.errors += error.item()

    def _after_epoch(self) -> None:
        self.size = dist.allreduce(self.size, reduction='sum')
        self.errors = dist.allreduce(self.errors, reduction='sum')
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, self.errors/self.size )

class MSE(Callback):
    def __init__(self,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'mse') -> None:
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor


    def _before_epoch(self):
        self.size = 0
        self.errors = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        error = torch.mean((outputs - targets) ** 2)
        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)
        #self.errors += error.item()

    def _after_epoch(self) -> None:
        self.size = dist.allreduce(self.size, reduction='sum')
        self.errors = dist.allreduce(self.errors, reduction='sum')
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, self.errors/self.size )

class MTE(Callback):
    def __init__(self,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'mte') -> None:
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor


    def _before_epoch(self):
        self.errors = 0
        self.size = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        error = torch.mean(torch.abs(outputs - targets) ** 4)
        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)

    def _after_epoch(self) -> None:
        self.size = dist.allreduce(self.size, reduction='sum')
        self.errors = dist.allreduce(self.errors, reduction='sum')
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, self.errors/self.size)


class MeanIoU(Callback):
    def __init__(self, 
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
    
    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)
    
    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        
        if type(outputs) != np.ndarray:
            for i in range(self.num_classes):
                self.total_seen[i] += torch.sum(targets == i).item()
                self.total_correct[i] += torch.sum(
                    (targets == i) & (outputs == targets)).item()
                self.total_positive[i] += torch.sum(
                    outputs == i).item()
        else:
            for i in range(self.num_classes):
                self.total_seen[i] += np.sum(targets == i)
                self.total_correct[i] += np.sum((targets == i)
                                                & (outputs == targets))
                self.total_positive[i] += np.sum(outputs == i)
    
    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i], reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i], reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i], reduction='sum')
        
        ious = []
        
        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] +
                    self.total_positive[i] -
                    self.total_correct[i])
                ious.append(cur_iou)
        
        miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
        else:
            print(ious)
            print(miou)
