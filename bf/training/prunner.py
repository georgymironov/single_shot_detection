from collections import defaultdict
import random
import re

import torch
import torch.nn as nn

from bf.utils import torch_utils


class Critetion(object):
    def __init__(self, model, include_paths=None):
        self.model = model

        if not include_paths:
            include_paths = []

        def _include(item):
            name, module = item
            return any(name.startswith(p) for p in include_paths) and isinstance(module, nn.Conv2d)

        self.modules = dict(filter(_include, model.named_modules()))

    def get_path(self):
        raise NotImplementedError

class RandomSampling(Critetion):
    def get_path(self):
        index = random.randint(0, sum(x.out_channels for x in self.modules.values()))
        cumsum = 0
        for name, module in self.modules.items():
            if cumsum + module.out_channels > index:
                return name, index - cumsum
            cumsum += module.out_channels


def _to_torch_path(onnx_node):
    return '.'.join(re.findall(r'\[(.*?)\]', onnx_node.scopeName()))

def _is_depthwise_conv_onnx(onnx_node):
    if onnx_node.kind() != 'onnx::Conv':
        return False
    return next(onnx_node.inputs()).type().sizes()[1] == onnx_node.output().type().sizes()[1] == onnx_node['group']

def _is_depthwise_conv(module):
    return module.out_channels == module.in_channels == module.groups

def _mask_parameter(parameter, mask):
    if parameter is not None:
        parameter.data = parameter.data[mask]
        if parameter.grad is not None:
            parameter.grad = parameter.grad[mask]

def _remove_conv_out_channel(module, index):
    mask = torch.ones(module.out_channels, dtype=torch.uint8)
    mask[index] = 0

    _mask_parameter(module.weight, mask)
    _mask_parameter(module.bias, mask)

    if _is_depthwise_conv(module):
        module.out_channels = module.in_channels = module.groups = module.out_channels - 1
    else:
        module.out_channels = module.out_channels - 1

def _remove_conv_in_channel(module, index):
    mask = torch.ones(module.in_channels, dtype=torch.uint8)
    mask[index] = 0

    if _is_depthwise_conv(module):
        _mask_parameter(module.weight, mask)
        _mask_parameter(module.bias, mask)
        module.out_channels = module.in_channels = module.groups = module.out_channels - 1
    else:
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

    def __init__(self, model, include_paths=None, criterion='RandomSampling'):
        self.model = model
        self.modules = dict(torch_utils.get_leaf_modules(model))
        self.criterion = globals().get(criterion)(model, include_paths)

        self.trace = torch_utils.get_onnx_trace(model)  # writing to self to prevent deallocation

        graph = self.trace.graph()
        self.nodes = dict()

        for node in graph.nodes():
            for output in node.outputs():
                self.nodes[output.unique()] = node

        self.node_paths = {_to_torch_path(x): x.output().unique() for x in graph.nodes() if x.kind() == 'onnx::Conv'}
        self.output_nodes = defaultdict(set)

        for node in graph.nodes():
            for inp in node.inputs():
                self.output_nodes[inp.unique()].add(node)

    def get_affected_nodes(self, unique, type_='out', memo=None):
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
            path, index = self.criterion.get_path()
        elif path is None or index is None:
            raise ValueError('Both "path" and "index" should be provided')

        print(f'Pruning: {path} #{index}')

        print('Affected nodes:')
        for path, channel_type in self.get_affected_nodes(self.node_paths[path], 'out'):
            module = self.modules[path]
            if isinstance(module, nn.Conv2d):
                if channel_type == 'in':
                    _remove_conv_in_channel(module, index)
                else:
                    _remove_conv_out_channel(module, index)
            elif isinstance(module, nn.BatchNorm2d):
                _remove_batchnorm_channel(module, index)
            else:
                raise TypeError(f'Unsupported layer type: {type(module)}')
            print(f'{path} #{index} {channel_type}')
