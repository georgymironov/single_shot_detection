import math
import random

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
        self.modules = {name: module for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)}
        self.connected = {k: v for k, v in trace_inspector.connected.items() if k in self.modules}

        if not include_paths:
            include_paths = []

        self.include_paths = include_paths
        self.filtered_modules = dict(filter(self._include, self.modules.items()))

    def _include(self, item):
        name, _ = item
        return any(name.startswith(p) for p in self.include_paths)

    def _share_connected(self, weights):
        weights = {
            name: torch.max(torch.cat([weights[x] for x in self.connected[name]], dim=1), dim=1)[0]
            for name in weights.keys()
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
        modules = {name: self.filtered_modules[name] for name in weights.keys()}

        weights = torch.cat(list(weights.values()))
        _, indexes = torch.topk(weights, num, largest=False)
        indexes = [x.item() for x in indexes]

        return _get_paths(modules, indexes)

    def get_path(self, num=1):
        raise NotImplementedError

class RandomSampling(Critetion):
    def get_path(self, num=1):
        total_channels = sum(x.out_channels for x in self.filtered_modules.values())
        indexes = [random.randint(0, total_channels) for _ in range(num)]
        return _get_paths(self.filtered_modules, indexes)

class MinL1Norm(Critetion):
    def get_path(self, num=1):
        norm = {
            name: module.weight.abs().sum(dim=(1, 2, 3)).unsqueeze(1)
                if module.out_channels > num
                else torch.full((module.out_channels, 1), math.inf, dtype=module.weight.dtype, device=module.weight.device)
            for name, module in self.filtered_modules.items()
        }
        return self._get_path_by_weights(norm, num)

class MinL2Norm(Critetion):
    def get_path(self, num=1):
        norm = {
            name: module.weight.pow(2).sum(dim=(1, 2, 3)).sqrt().unsqueeze(1)
                if module.out_channels > num
                else torch.full((module.out_channels, 1), math.inf, dtype=module.weight.dtype, device=module.weight.device)
            for name, module in self.filtered_modules.items()
        }
        return self._get_path_by_weights(norm, num)
