from typing import Any, Callable, Dict
import numpy as np
import os
import torch
from torch import nn
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
#from modules.peak_stimulator import peak_stimulator, prm_backpropagation

__all__ = ['SemanticKITTITrainer']


class SemanticKITTITrainer(Trainer):
    def __init__(self, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler,
                 num_workers: int, seed: int, out_save_dir: str,
                 tfevent:str=None, tfeventname:str=None) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.epoch_num = 1
        self.out_save_dir = out_save_dir
        self.tfevent = tfevent
        self.tfeventname = tfeventname

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num-1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
                self.seed + (self.epoch_num-1) * self.num_workers + worker_id)
        print ("lr: ", self.optimizer.state_dict()['param_groups'][0]['lr'] )

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:   
        _inputs = dict()
        for key, value in feed_dict.items():
            if key not in ['subsize', 'pc_file','file_name','calibs','labels','rot_mat', 'scale_factor']:
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar'] # voxelized input, .C is point cloud (N,4)
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
            """
            #convertion from voxelized data to original size
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs = []
            _targets = []
            for idx in range(invs.C[:, -1].max()+1):
                cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][
                    cur_inv]# .argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)                
            #print(torch.min(outputs.cpu()), torch.max(outputs.cpu()))
            """

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        
    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:

        # Before step, generetate new CRM with required radius.
        # Then update feed_dict
        from core.datasets.utils import generate_CRM_wfiles
        _inputs = dict()
        for key, value in feed_dict.items():
            if key not in ['subsize', 'pc_file','file_name','calibs','labels','rot_mat', 'scale_factor']:
                _inputs[key] = value.cuda()
        #inputs = _inputs['lidar'] # voxelized input, .C is point cloud (N,4)
        points = _inputs['lidar'].F.cpu()
        calibs = feed_dict['calibs']
        labels = feed_dict['labels']
        rot_mat = feed_dict['rot_mat']
        scale_factor = feed_dict['scale_factor']
        subsizes = feed_dict['subsize']
        start = 0
        for i in range(len(calibs)):
            calib = calibs[i]
            label = labels[i]
            rot_matrix = rot_mat[i]
            scale_fac = scale_factor[i]
            subsize = subsizes[i]
            
            point = np.asarray(points[start:subsize+start, :]).astype(np.float32)
            radius = int ((self.num_epochs-self.epoch_num) / 5)
            if radius<2:
                radius=2
            crm_target = generate_CRM_wfiles(radius, points = point, labels_path=label,
                                             calibs_path=calib, rot_mat=rot_matrix, scale_factor=scale_fac)
            
            feed_dict['targets'].F[start:subsize+start, :] = torch.from_numpy(crm_target).to(feed_dict['targets'].F)
            start += subsize

    def _after_step(self, output_dict: Dict[str, Any]):
        from torch.utils.tensorboard import SummaryWriter

        assert  self.tfevent==None or self.tfeventname==None

        writer = SummaryWriter(self.tfevent+self.tfeventname)
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param.grad, self.global_step)