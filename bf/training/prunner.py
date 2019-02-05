from collections import defaultdict
import logging
import math
import random
import re

import torch
import torch.nn as nn

from bf.utils import torch_utils


class Critetion(object):
    def __init__(self, modules, connected=None, include_paths=None):
        self.modules = {name: module for name, module in modules.items() if isinstance(module, nn.Conv2d)}
        self.connected = connected or {}

        if not include_paths:
            include_paths = []

        self.include_paths = include_paths
        self.filtered_modules = dict(filter(self._include, self.modules.items()))

    def _include(self, item):
        name, _ = item
        return any(name.startswith(p) for p in self.include_paths)

    def _get_path(self, indexes, num=1):
        indexes = sorted(indexes)
        cumsum = 0
        i = 0
        for name, module in self.filtered_modules.items():
            while cumsum + module.out_channels > indexes[i]:
                yield name, indexes[i] - cumsum
                i += 1
                if num == i:
                    return
            cumsum += module.out_channels

    def get_path(self, num=1):
        raise NotImplementedError

class RandomSampling(Critetion):
    def get_path(self, num=1):
        total_channels = sum(x.out_channels for x in self.filtered_modules.values())
        indexes = [random.randint(0, total_channels) for _ in range(num)]
        yield from self._get_path(indexes, num)

class MinL1Norm(Critetion):
    def get_path(self, num=1):
        norm = {
            name: module.weight.abs().sum(dim=(1, 2, 3)).unsqueeze(1)
                if module.out_channels > 1
                else torch.tensor([[math.inf]], dtype=module.weight.dtype, device=module.weight.device)
            for name, module in self.modules.items()
        }
        norm = {
            name: torch.max(torch.cat([norm[x] for x in self.connected[name]], dim=1), dim=1)[0]
            for name in norm.keys()
        }
        norm = dict(filter(self._include, norm.items()))
        norm = torch.cat([x for x in norm.values()])
        _, indexes = torch.topk(norm, num, largest=False)

        yield from self._get_path(indexes, num)

def _to_torch_path(onnx_node):
    return '.'.join(re.findall(r'\[(.*?)\]', onnx_node.scopeName()))

def _is_depthwise_conv_onnx(onnx_node):
    if onnx_node.kind() != 'onnx::Conv':
        return False
    return next(onnx_node.inputs()).type().sizes()[1] == onnx_node.output().type().sizes()[1] == onnx_node['group']

def _is_depthwise_conv(module):
    return isinstance(module, nn.Conv2d) and module.out_channels == module.in_channels == module.groups

def _mask_parameter(parameter, mask):
    if parameter is not None:
        parameter.data = parameter.data[mask]
        if parameter.grad is not None:
            parameter.grad = parameter.grad[mask]

def _remove_depthwise_conv_channel(module, index):
    mask = torch.ones(module.out_channels, dtype=torch.uint8)
    mask[index] = 0

    _mask_parameter(module.weight, mask)
    _mask_parameter(module.bias, mask)

    module.out_channels = module.in_channels = module.groups = module.out_channels - 1

def _remove_conv_out_channel(module, index):
    mask = torch.ones(module.out_channels, dtype=torch.uint8)
    mask[index] = 0

    _mask_parameter(module.weight, mask)
    _mask_parameter(module.bias, mask)

    module.out_channels = module.out_channels - 1

def _remove_conv_in_channel(module, index):
    mask = torch.ones(module.in_channels, dtype=torch.uint8)
    mask[index] = 0

    module.weight.data = module.weight.data[:, mask]
    if module.weight.grad is not None:
        module.weight.grad = module.weight.grad[:, mask]

    module.in_channels = module.in_channels - 1

def _remove_batchnorm_channel(module, index):
    mask = torch.ones(module.num_features, dtype=torch.uint8)
    mask[index] = 0

    _mask_parameter(module.weight, mask)
    _mask_parameter(module.bias, mask)
    _mask_parameter(module.running_mean, mask)
    _mask_parameter(module.running_var, mask)

    module.num_features = module.num_features - 1

class Prunner(object):
    _affected_in_node_types = ['onnx::Conv', 'onnx::BatchNormalization']
    _affected_out_node_types = ['onnx::Conv']

    def __init__(self, model, include_paths=None, criterion='RandomSampling', num=1):
        self.num = num
        self.model = model
        self.modules = dict(torch_utils.get_leaf_modules(model))

        self.trace = torch_utils.get_onnx_trace(model)  # writing to self to prevent deallocation

        graph = self.trace.graph()

        self.output_nodes = defaultdict(set)
        for node in graph.nodes():
            for inp in node.inputs():
                self.output_nodes[inp.unique()].add(node)

        nodes = {}
        self.ignore = []
        for node in graph.nodes():
            path = _to_torch_path(node)
            if path not in nodes:
                nodes[path] = node
                continue

            for output in node.outputs():
                for affected in self.output_nodes[output.unique()]:
                    for old_output, new_output in zip(node.outputs(), nodes[path].outputs()):
                        affected.replaceInputWith(old_output, new_output)
                self.ignore.append(output.unique())

        self.nodes = {}
        for node in graph.nodes():
            for output in node.outputs():
                self.nodes[output.unique()] = node

        self.output_nodes = defaultdict(set)
        for node in graph.nodes():
            for inp in node.inputs():
                self.output_nodes[inp.unique()].add(node)

        self.node_paths = {_to_torch_path(x): x.output().unique()
                           for x in graph.nodes()
                           if x.kind() == 'onnx::Conv' and x.output().unique() not in self.ignore}

        self.affected = {k: list(self.get_affected_nodes(v, 'out')) for k, v in self.node_paths.items()}

        connected = {k: [x[0] for x in v
                         if isinstance(self.modules[x[0]], nn.Conv2d) and x[1] == 'out']
                     for k, v in self.affected.items()}

        self.criterion = globals().get(criterion)(dict(model.named_modules()), connected, include_paths)

    def get_affected_nodes(self, unique, type_='out', memo=None):
        if unique in self.ignore:
            return

        if memo is None:
            memo = set()
        node = self.nodes[unique]
        affected = _to_torch_path(node), type_

        if affected in memo:
            return

        def _get_affected_in_nodes():
            for output_node in self.output_nodes[unique]:
                for output in output_node.outputs():
                    yield from self.get_affected_nodes(output.unique(), 'in', memo)

        def _get_affected_out_nodes():
            for inp in node.inputs():
                if inp.unique() in self.nodes:
                    yield from self.get_affected_nodes(inp.unique(), 'out', memo)

        if _is_depthwise_conv_onnx(node):
            yield affected
            memo.add((_to_torch_path(node), 'in'))
            memo.add((_to_torch_path(node), 'out'))
            yield from _get_affected_in_nodes()
            yield from _get_affected_out_nodes()
        elif type_ == 'in':
            if node.kind() in self._affected_in_node_types and type_ == 'in':
                yield affected
                memo.add(affected)
            if node.kind() != 'onnx::Conv':
                yield from _get_affected_in_nodes()
            if node.kind() == 'onnx::Add':
                yield from _get_affected_out_nodes()
        else:
            if node.kind() in self._affected_out_node_types and type_ == 'out':
                yield affected
                memo.add(affected)
            if node.kind() == 'onnx::Conv':
                yield from _get_affected_in_nodes()
            else:
                yield from _get_affected_out_nodes()

    def prune(self, path=None, index=None):
        if path is None and index is None:
            paths = self.criterion.get_path(self.num)
        elif path is None or index is None:
            raise ValueError('Both "path" and "index" should be provided')

        logging.info('Pruning:')
        for path, index in paths:
            logging.info(f'{path} #{index}')
            logging.debug('Affected nodes:')
            for path, channel_type in self.affected[path]:
                module = self.modules[path]
                if _is_depthwise_conv(module):
                    _remove_depthwise_conv_channel(module, index)
                elif isinstance(module, nn.Conv2d):
                    if channel_type == 'in':
                        _remove_conv_in_channel(module, index)
                    else:
                        _remove_conv_out_channel(module, index)
                elif isinstance(module, nn.BatchNorm2d):
                    _remove_batchnorm_channel(module, index)
                else:
                    raise TypeError(f'Unsupported layer type: {type(module)}')
                logging.debug(f'{path} #{index} {channel_type}')
