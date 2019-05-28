import logging
import math
import random
from itertools import chain

import torch
import torch.nn as nn


def _get_paths(modules, indexes):
    indexes = sorted(indexes)
    cumsum = 0
    i = 0
    for name, module in modules.items():
        while cumsum + module.out_channels > indexes[i]:
            yield name, indexes[i] - cumsum
            i += 1
            if i == len(indexes):
                return
        cumsum += module.out_channels


class Critetion(object):
    def __init__(self, trace_inspector, include_paths=None):
        self.trace_inspector = trace_inspector
        self.model = trace_inspector.model

        self.included_modules = {name: module for name, module in self.model.named_modules()
                                 if isinstance(module, nn.Conv2d) and (include_paths is None or any(name.startswith(p) for p in include_paths))}
        self.connected = {k: v for k, v in trace_inspector.connected.items() if k in self.included_modules}
        self.modules = {name: module for name, module in self.model.named_modules() if name in set(chain(*self.connected.values()))}

    def _share_connected(self, weights):
        weights = {
            name: torch.max(torch.cat([weights[x] for x in self.connected[name]], dim=1), dim=1)[0]
            for name in self.included_modules.keys()
        }
        processed = set()
        for name in list(weights.keys()):
            for connected in self.connected[name]:
                if connected != name and connected not in processed and connected in weights:
                    del weights[connected]
                    processed.add(name)
        return weights

    def _get_path_by_weights(self, weights, num):
        weights = self._share_connected(weights)
        modules = {name: self.included_modules[name] for name in weights.keys()}

        for name, weight in weights.items():
            assert self.included_modules[name].weight.size(0) == weight.size(0)

        weights = torch.cat(list(weights.values()))
        _, indexes = torch.topk(weights, num, largest=False)
        indexes = [x.item() for x in indexes]

        return _get_paths(modules, indexes)

    def get_path(self, num=1):
        raise NotImplementedError

class RandomSampling(Critetion):
    def get_path(self, num=1):
        total_channels = sum(x.out_channels for x in self.included_modules.values())
        indexes = [random.randint(0, total_channels) for _ in range(num)]
        return _get_paths(self.included_modules, indexes)

class MinL1Norm(Critetion):
    def get_path(self, num=1):
        norm = {
            name: module.weight.abs().sum(dim=(1, 2, 3)).unsqueeze(1)
                if module.out_channels > num
                else torch.full((module.out_channels, 1), math.inf, dtype=module.weight.dtype, device=module.weight.device)
            for name, module in self.modules.items()
        }
        return self._get_path_by_weights(norm, num)

class MinL2Norm(Critetion):
    def get_path(self, num=1):
        norm = {
            name: module.weight.pow(2).sum(dim=(1, 2, 3)).sqrt().unsqueeze(1)
                if module.out_channels > num
                else torch.full((module.out_channels, 1), math.inf, dtype=module.weight.dtype, device=module.weight.device)
            for name, module in self.modules.items()
        }
        return self._get_path_by_weights(norm, num)

class _mean_activation_hook:
    def __init__(self, mean_activation):
        self.num_steps = 0
        self.mean_activation = mean_activation

    def __call__(self, module, input, output):
        with torch.no_grad():
            self.mean_activation *= self.num_steps / (self.num_steps + 1)
            self.mean_activation += output.mean(dim=(0, 2, 3)) / (self.num_steps + 1)
        self.num_steps += 1

class MinActivation(Critetion):
    def __init__(self, *args, **kwargs):
        super(MinActivation, self).__init__(*args, **kwargs)

        self.activation_map = {}
        for name in self.included_modules.keys():
            activations = set(self.trace_inspector.get_descendent_of_type(name,
                                                                          types=self.trace_inspector.activation_types,
                                                                          stop_on=['onnx::Conv']))
            assert len(activations) <= 1

            if not activations:
                logging.warn(f'WRN: Layer "{name}" does not have an activation')
            else:
                self.activation_map[activations.pop()] = name

        for name, module in self.model.named_modules():
            if name in self.activation_map:
                conv = self.modules[self.activation_map[name]]
                conv.register_buffer('mean_activation', torch.zeros((conv.out_channels,), dtype=conv.weight.dtype, device=conv.weight.device))
                module.register_forward_hook(_mean_activation_hook(conv.mean_activation))

    def get_path(self, num=1):
        activation = {
            name: module.mean_activation.unsqueeze(1)
                if module.out_channels > num
                else torch.full((module.out_channels, 1), math.inf, dtype=module.weight.dtype, device=module.weight.device)
            for name, module in self.modules.items()
            if hasattr(module, 'mean_activation') and module.mean_activation.sum().ne(0)
        }

        if not activation:
            return []

        return self._get_path_by_weights(activation, num)
