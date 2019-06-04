from collections import defaultdict
import logging

import torch
import torch.nn as nn

from . import criterions
from .trace_inspector import TraceInspector
from bf.utils import torch_utils


def _is_depthwise_conv(module):
    return isinstance(module, nn.Conv2d) and module.out_channels == module.in_channels == module.groups

def _mask_parameter(module, name, mask):
    parameter = getattr(module, name, None)
    if parameter is not None:
        parameter.data = parameter.data[mask]
        if parameter.grad is not None:
            parameter.grad = parameter.grad[mask]

def _remove_depthwise_conv_channel(module, indexes):
    mask = torch.ones(module.out_channels, dtype=torch.uint8)
    mask[indexes] = 0

    _mask_parameter(module, 'weight', mask)
    _mask_parameter(module, 'bias', mask)
    _mask_parameter(module, 'mean_activation', mask)

    module.out_channels = module.in_channels = module.groups = module.out_channels - len(indexes)

def _remove_conv_out_channel(module, indexes):
    mask = torch.ones(module.out_channels, dtype=torch.uint8)
    mask[indexes] = 0

    _mask_parameter(module, 'weight', mask)
    _mask_parameter(module, 'bias', mask)
    _mask_parameter(module, 'mean_activation', mask)

    module.out_channels = module.out_channels - len(indexes)

def _remove_conv_in_channel(module, indexes):
    mask = torch.ones(module.in_channels, dtype=torch.uint8)
    mask[indexes] = 0

    module.weight.data = module.weight.data[:, mask]
    if module.weight.grad is not None:
        module.weight.grad = module.weight.grad[:, mask]

    module.in_channels = module.in_channels - len(indexes)

def _remove_batchnorm_channel(module, indexes):
    mask = torch.ones(module.num_features, dtype=torch.uint8)
    mask[indexes] = 0

    _mask_parameter(module, 'weight', mask)
    _mask_parameter(module, 'bias', mask)
    _mask_parameter(module, 'running_mean', mask)
    _mask_parameter(module, 'running_var', mask)

    module.num_features = module.num_features - len(indexes)

class Pruner(object):
    def __init__(self, model, include_paths=None, criterion='MinL1Norm', num=1):
        self.num = num
        self.model = model
        self.modules = dict(torch_utils.get_leaf_modules(model))
        self.trace_inspector = TraceInspector(model)
        self.criterion = getattr(criterions, criterion)(self.trace_inspector, include_paths)

    def prune(self, paths=None):
        if not paths:
            paths = self.criterion.get_path(self.num)

        logging.info('Pruning:')

        if not paths:
            logging.info('Nothing!')
            return

        grouped_paths = defaultdict(set)
        for path, index in paths:
            logging.info(f'{path} #{index}')
            logging.debug('-' * 25)
            for path, channel_type, index in self.trace_inspector.get_affected_nodes(path, index):
                grouped_paths[(path, channel_type)].add(index)
                logging.debug(f'{path} #{index} {channel_type}')
            logging.debug('-' * 25)

        for (path, channel_type), indexes in grouped_paths.items():
            indexes = list(indexes)
            module = self.modules[path]
            if _is_depthwise_conv(module):
                _remove_depthwise_conv_channel(module, indexes)
            elif isinstance(module, nn.Conv2d):
                if channel_type == 'in':
                    _remove_conv_in_channel(module, indexes)
                else:
                    _remove_conv_out_channel(module, indexes)
            elif isinstance(module, nn.BatchNorm2d):
                _remove_batchnorm_channel(module, indexes)
            else:
                raise TypeError(f'Unsupported layer type: {type(module)}')
            logging.debug(f'{(path, channel_type)} {indexes}')
