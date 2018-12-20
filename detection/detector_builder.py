import functools

import torch
import torch.nn as nn

from bf.modules import conv
from detection.detector import Detector
from detection.modules.anchor import AnchorGenerator


def _get_leaf_modules(module, memo=None, prefix=''):
    if memo is None:
        memo = set()
    if module not in memo:
        memo.add(module)
        if not module._modules:
            yield prefix, module
        else:
            for name, submodule in module._modules.items():
                if submodule is None:
                    continue
                submodule_prefix = prefix + ('_' if prefix else '') + name
                for m in _get_leaf_modules(submodule, memo, submodule_prefix):
                    yield m


class DetectorBuilder(object):
    def __init__(self,
                 base,
                 num_classes,
                 source_layers,
                 extra_layer_depth=(None, None, 512, 256, 256, 256),
                 num_scales=6,
                 sizes=None,
                 min_scale=None,
                 max_scale=None,
                 aspect_ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 0.3333]] * 3 + [[1, 2, 0.5]] * 2,
                 steps=None,
                 offsets=[0.5, 0.5],
                 num_branches=None,
                 last_feature_layer=None,
                 depth_multiplier=1.0,
                 use_depthwise=False,
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 batch_norm={}):

        assert isinstance(base, nn.Module)
        assert isinstance(base.features, nn.Sequential)
        assert len(aspect_ratios) == num_scales
        assert len(source_layers) == num_scales
        assert len(extra_layer_depth) == num_scales
        assert sizes is not None or (min_scale is not None and max_scale is not None)

        if steps is not None:
            assert len(steps) == num_scales
            self.steps = steps
        else:
            self.steps = [None] * num_scales

        if num_branches is None:
            num_branches = [1] * self.num_scales
        else:
            assert len(num_branches) == num_scales

        if min_scale is not None and max_scale is not None:
            self.scales = torch.linspace(min_scale, max_scale, num_scales + 1)
            print(f'Detector (Scales: {self.scales[:-1]})')
        else:
            self.scales = None

        if sizes is not None:
            self.sizes = sizes

        self.base = base
        self.num_classes = num_classes
        self.use_depthwise = use_depthwise
        self.depth_multiplier = depth_multiplier
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.num_branches = num_branches
        self.source_layers = source_layers
        self.extra_layer_depth = extra_layer_depth
        self.last_feature_layer = last_feature_layer
        self.activation_params = activation
        self.activation = functools.partial(getattr(nn, activation['name']), **activation['args'])
        self.batch_norm_params = batch_norm
        self.num_boxes = [(len(ar) + 1) * b for ar, b in zip(aspect_ratios, num_branches)]
        self.source_out_channels = self.get_source_out_channels()

    def build(self):
        feature_layers = list(self.base.features.children())
        if self.last_feature_layer is not None:
            feature_layers = feature_layers[:(self.last_feature_layer + 1)]
        features = nn.Sequential(*feature_layers)
        extras = self.get_extras()
        heads = self.get_heads()
        priors = self.get_priors()

        return Detector(self.num_classes,
                        features,
                        extras,
                        heads,
                        priors,
                        self.source_layers)

    def get_source_out_channels(self):
        source_out_channels = []

        x = torch.ones((1, 3, 300, 300), dtype=torch.float)
        source_layer_idx = 0
        for i, layer in enumerate(self.base.features):
            source_layer = self.source_layers[source_layer_idx]
            x = layer(x)
            if isinstance(source_layer, int):
                if i == source_layer:
                    source_out_channels.append(x.size(1))
                    source_layer_idx += 1
            elif isinstance(source_layer, tuple):
                if i == source_layer[0]:
                    y = x
                    for name, child in self.base.features[i + 1].named_children():
                        y = child(y)
                        if name == source_layer[1]:
                            break
                    else:
                        raise ValueError(f'Wrong layer {source_layer}')
                    source_out_channels.append(y.size(1))
                    source_layer_idx += 1

        return source_out_channels

    def get_extras(self):
        extras = nn.ModuleList()
        in_channels = self.source_out_channels[-1]

        for source_layer, depth in zip(self.source_layers, self.extra_layer_depth):
            if depth is None:
                continue

            out_channels = int(depth * self.depth_multiplier)
            layers = []

            layers.append(conv.Conv2dBn(in_channels, out_channels // 2, kernel_size=1, bias=False,
                                        activation_params=self.activation_params,
                                        use_bn=True, batch_norm_params=self.batch_norm_params))

            in_channels = out_channels // 2

            if self.use_depthwise:
                if source_layer == 's':
                    layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, stride=2,
                                                         padding=1, bias=False, activation_params=self.activation_params,
                                                         use_bn=True, batch_norm_params=self.batch_norm_params))
                elif source_layer == '':
                    layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                                         activation_params=self.activation_params,
                                                         use_bn=True, batch_norm_params=self.batch_norm_params))
            else:
                if source_layer == 's':
                    layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, stride=2,
                                                padding=1, bias=False, activation_params=self.activation_params,
                                                use_bn=True, batch_norm_params=self.batch_norm_params))
                elif source_layer == '':
                    layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                                activation_params=self.activation_params,
                                                use_bn=True, batch_norm_params=self.batch_norm_params))

            self.source_out_channels.append(out_channels)
            extras.append(nn.Sequential(*layers))
            in_channels = out_channels

        return extras

    def get_heads(self):
        heads = nn.ModuleList()
        for in_channels, num_boxes in zip(self.source_out_channels, self.num_boxes):
            heads.append(nn.ModuleDict({
                'class': nn.Conv2d(in_channels, num_boxes * self.num_classes, kernel_size=3, padding=1),
                'loc': nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, padding=1)
            }))
        return heads

    def get_priors(self):
        priors = []
        for i, (ratios, step, num_branches) in enumerate(zip(self.aspect_ratios, self.steps, self.num_branches)):
            if self.scales is not None:
                priors.append(AnchorGenerator(ratios,
                                              min_scale=self.scales[i],
                                              max_scale=self.scales[i + 1],
                                              step=step,
                                              num_branches=num_branches))
            else:
                priors.append(AnchorGenerator(ratios,
                                              min_size=self.sizes[i],
                                              max_size=self.sizes[i + 1],
                                              step=step,
                                              num_branches=num_branches))
        return priors
