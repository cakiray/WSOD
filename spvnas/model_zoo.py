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
from core.models.semantic_kitti.spvcnn import SPVCNN
from core.models.semantic_kitti.spvnas_cnn import SPVNAS_CNN


__all__ = ['spvnas_specialized', 'spvcnn', 'spvnas_cnn']


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

def spvnas_cnn(pretrained=False, **kwargs):

    input_channels = kwargs.get('input_channels', 5)
    num_classes = kwargs.get('num_classes', 1)
    model = SPVNAS_CNN(
        num_classes=num_classes,
        input_channels = input_channels,
        macro_depth_constraint=1,
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        dict_ = torch.load(kwargs['weights'])['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)
        print("Model weights are loaded.")

    return model

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

def spvnas_cnn_specialized(pretrained=False, **kwargs):

    input_channels = kwargs.get('input_channels', 4)
    num_classes = kwargs.get('num_classes', 1)
    model = SPVNAS_CNN(
        num_classes=19,
        input_channels = input_channels,
        macro_depth_constraint=1,
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        dict_ = torch.load(kwargs['weights'])['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)
        print("Model weights are loaded.")

    model = model.change_last_layer(num_classes=num_classes)
    return model

def spvnas_specialized(net_id, pretrained=True,  **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_specialized/'
    net_config = json.load(open(
        download_url(url_base + net_id + '/net.config', model_dir='.torch/spvnas_specialized/%s/' % net_id)
    ))
    input_channels = kwargs.get('input_channels', 4)
    model = SPVNAS(
        num_classes=net_config['num_classes'],
        input_channels = input_channels,
        macro_depth_constraint=1,
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')
    model.manual_select(net_config)
    model = model.determinize()
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
    #input_channels = kwargs.get('input_channels', 4)
    #num_classes = kwargs.get('num_classes', net_config['num_classes'])
    num_classes = net_config['num_classes']
    input_channels = 4
    model = SPVCNN(
        num_classes= num_classes,
        input_channels=input_channels,
        cr=net_config['cr'],
        pres=net_config['pres'],
        vres=net_config['vres']
    ).to('cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu')
    #model = model.change_last_layer(num_classes)
    if pretrained:
        if kwargs['weights'] is not None:
            dict_ = torch.load(kwargs['weights'])['model']
            init = dict()
            for key in dict_:
                init[key.replace('module.','')] = dict_[key]
        else:
            init = torch.load(
            download_url(url_base + net_id + '/init', model_dir='.torch/spvcnn/%s/' % net_id),
            map_location='cuda:%d'%dist.local_rank() if torch.cuda.is_available() else 'cpu'
        )['model']
        
        model.load_state_dict(init)
     
    model = model.change_last_layer(num_classes=1)
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
    model = model.determinize()

    if pretrained:
        dict_ = torch.load(kwargs['weights'])['model']
        dict_correct_naming = dict()
        for key in dict_:
            dict_correct_naming[key.replace('module.','')] = dict_[key]
        model.load_state_dict(dict_correct_naming)

    return model

