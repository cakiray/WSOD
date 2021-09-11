import os
import sys
import json
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from collections import OrderedDict

import torch
import torch.nn as nn

from torchpack import distributed as dist

from core.models.semantic_kitti.spvnas import SPVNAS
from core.models.semantic_kitti.minkunet import MinkUNet
from core.models.semantic_kitti.spvcnn import SPVCNN
from core.models.semantic_kitti.spvnas_cnn import SPVNAS_CNN


__all__ = ['spvnas_specialized', 'minkunet', 'spvcnn']


def download_url(url, model_dir='~/.torch/', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    cached_file = model_dir
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file

def spvnas_best(net_id, weights, configs, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvnas_specialized/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4)
    model = SPVNAS(
        configs.data.num_classes,
        input_channels = input_channels,
        macro_depth_constraint=1
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')
    model.manual_select(net_config)
    model = model.determinize()
    dict_ = torch.load(weights)['model']
    dict_correct_naming = dict()
    for key in dict_:
        dict_correct_naming[key.replace('module.','')] = dict_[key]
    model.load_state_dict(dict_correct_naming)
    return model

def spvnas_cnn(pretrained=False, **kwargs):

    input_channels = kwargs.get('input_channels', 5)
    num_classes = kwargs.get('num_classes', 1)
    model = SPVNAS_CNN(
        num_classes=num_classes,
        input_channels = input_channels,
        macro_depth_constraint=1,
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')
    #model.shrink()
    #model.remove_skipconnection()
    if pretrained:
        dict_ = torch.load(kwargs['weights'])['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)

    return model

def spvnas_specialized(net_id, pretrained=True,  **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvnas_specialized/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4)
    model = SPVNAS(
        net_config['num_classes'],
        input_channels = input_channels,
        macro_depth_constraint=1,
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')
    model.manual_select(net_config)
    model = model.determinize()
    model.shrink()
    if pretrained:
        
        init = torch.load(
            download_url(url_base + net_id + '/init', model_dir='.torch/spvnas_specialized/%s/' % net_id),
            map_location='cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu'
        )['model']
        model.load_state_dict(init)
       
        
    return model


def spvnas_supernet(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_supernet/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvnas_supernet/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4 )

    model = SPVNAS(
        net_config['num_classes'],
        input_channels = input_channels,
        macro_depth_constraint=net_config['macro_depth_constraint'],
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        init = torch.load(
            download_url(url_base + net_id + '/init', model_dir='.torch/spvnas_supernet/%s/' % net_id),
            map_location='cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu'
        )['model']
        model.load_state_dict(init)
    return model


def minkunet(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/minkunet/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/minkunet/%s/' % net_id)
    ))

    model = MinkUNet(
        num_classes=net_config['num_classes'],
        cr=net_config['cr']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        init = torch.load(
            download_url(url_base + net_id + '/init', model_dir='.torch/minkunet/%s/' % net_id),
            map_location='cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu'
        )['model']
        model.load_state_dict(init)
    return model

def spvcnn_best(net_id, weights, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvcnn/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvcnn/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4)
    num_classes = kwargs.get('num_classes', net_config['num_classes'])
    model = SPVCNN(
        num_classes= num_classes,
        input_channels=input_channels,
        cr=net_config['cr'],
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        dict_ = torch.load(weights)['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)

    return model

def spvcnn(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvcnn/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvcnn/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4)
    num_classes = kwargs.get('num_classes', net_config['num_classes'])
    model = SPVCNN(
        num_classes= num_classes,
        input_channels=input_channels,
        cr=net_config['cr'],
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        init = torch.load(
            download_url(url_base + net_id + '/init', model_dir='.torch/spvcnn/%s/' % net_id),
            map_location='cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu'
        )['model']
        model.load_state_dict(init)
    return model

def spvcnn_specialized(net_id, pretrained=True,  **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvnas_specialized/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4)
    model = SPVNAS(
        net_config['num_classes'],
        input_channels = input_channels,
        macro_depth_constraint=1,
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    model.manual_select(net_config)
    print("next model ", model)
    model = model.determinize()

    if pretrained:

        dict_ = torch.load(kwargs['weights'])['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)


    return model

def myspvcnn(configs, pretrained = False, weights=None, **kwargs):

    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    model = SPVCNN(
        input_channels= configs.data.input_channels,
        num_classes= configs.data.num_classes,
        cr=cr,
        pres=configs.dataset.voxel_size,
        vres=configs.dataset.voxel_size
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        dict_ = torch.load(weights)['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)

    return model
