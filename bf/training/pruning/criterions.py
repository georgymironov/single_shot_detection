import logging
import math
import random
from functools import partial
from itertools import chain

import torch
import torch.nn as nn

from ._hooks import _mean_activation_hook, _taylor_expansion_hook, _save_output_hook


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

        self.included_modules = {name: module for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)}

        if include_paths:
            self.included_modules = {name: module for name, module in self.included_modules.items()
                                     if any(name.startswith(p) for p in include_paths)}

        self.connected = {k: v for k, v in trace_inspector.connected.items() if k in self.included_modules}
        self.modules = {name: module for name, module in self.model.named_modules() if name in set(chain(*self.connected.values()))}

        self.concat_groups = {}
        for group in trace_inspector.concat_groups:
            for x in group:
                assert x not in self.concat_groups
                self.concat_groups[x] = group

    def _share_connected(self, weights):
        # ToDo: fix this case:
        #  conv    conv  conv
        #    |       |     |
        #    |        -add-
        #    |          |
        #     --concat--
        #          |

        _weights = {}
        processed = set()

        for name in self.included_modules.keys():
            if name in processed:
                continue
            connected = []
            for connection in self.connected[name]:
                if connection in processed or connection not in weights:
                    continue
                if connection in self.concat_groups:
                    group = self.concat_groups[connection]
                    connected.append(torch.cat([weights[x] for x in group], dim=0))
                    assert not processed.intersection(group)
                    processed.update(group)
                else:
                    connected.append(weights[connection])
                    processed.add(connection)
            if connected:
                _weights[name] = torch.max(torch.cat(connected, dim=1), dim=1)[0]

        return _weights

    def _exclude_last_layer(self, weights, num):
        for name, weight in weights.items():
            if self.modules[name].out_channels <= num:
                weight[weight.argmax()] = math.inf

    def _get_path_by_weights(self, weights, num):
        if not weights:
            return []

        self._exclude_last_layer(weights, num)

        weights = self._share_connected(weights)
        modules = {name: self.included_modules[name] for name in weights.keys()}

        for name, weight in weights.items():
            assert self.included_modules[name].weight.size(0) == weight.size(0)

        weights = torch.cat(list(weights.values()))
        values, indexes = torch.topk(weights, num, largest=False)
        logging.info(f'Pruned weights:')
        logging.info(values)
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
            name: module.weight.abs().sum(dim=(1, 2, 3)).view(-1, 1)
            for name, module in self.modules.items()
        }
        return self._get_path_by_weights(norm, num)

class MinL2Norm(Critetion):
    def get_path(self, num=1):
        norm = {
            name: module.weight.pow(2).sum(dim=(1, 2, 3)).sqrt().view(-1, 1)
            for name, module in self.modules.items()
        }
        return self._get_path_by_weights(norm, num)

class _Activation(Critetion):
    forward_hook = None
    backward_hook = None

    def __init__(self, *args, **kwargs):
        super(_Activation, self).__init__(*args, **kwargs)

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
                module.register_forward_hook(_save_output_hook)
                conv = self.modules[self.activation_map[name]]
                if self.forward_hook:
                    module.register_forward_hook(self.forward_hook(conv))
                if self.backward_hook:
                    module.register_backward_hook(self.backward_hook(conv))

    def get_path(self, num=1):
        weights = {
            name: module.pruning_criterion.view(-1, 1)
            for name, module in self.modules.items()
            if hasattr(module, 'pruning_criterion')
        }
        return self._get_path_by_weights(weights, num)

class MeanActivation(_Activation):
    def __init__(self, *args, momentum=0.9, **kwargs):
        self.forward_hook = partial(_mean_activation_hook, momentum=momentum)
        super(MeanActivation, self).__init__(*args, **kwargs)

# ref: https://arxiv.org/pdf/1611.06440.pdf
class TaylorExpansion(_Activation):
    def __init__(self, *args, momentum=0.9, **kwargs):
        self.backward_hook = partial(_taylor_expansion_hook, momentum=momentum)
        super(TaylorExpansion, self).__init__(*args, **kwargs)
