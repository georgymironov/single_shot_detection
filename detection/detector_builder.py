import functools

import torch.nn as nn

from bf.modules import conv
from bf.modules import features as _features
from bf.utils.misc_utils import get_ctor
from detection.detector import Detector

import detection.retina_net
import detection.ssd


def build(base,
          anchor_generator_params,
          num_classes,
          features,
          use_depthwise=False,
          extras={},
          predictor={}):

    extra_layers = extras.get('layers', [])

    # backward compatibility
    # ToDo: remove
    source_layers = features['out_layers']

    Features = get_ctor(_features, features['name'])
    features = Features(base, use_depthwise=use_depthwise, **features).eval()

    num_scales = features.num_outputs + len(extra_layers)
    source_out_channels = features.get_out_channels()

    anchor_generator = getattr(detection, anchor_generator_params['type']).anchor_generator
    priors = anchor_generator.get_priors(**anchor_generator_params)

    assert num_scales == len(priors)
    num_boxes = [x.num_boxes for x in priors]

    extras = get_extras(source_out_channels, use_depthwise=use_depthwise, **extras)
    predictor, heads = get_predictor(source_out_channels, num_boxes, num_classes, use_depthwise, **predictor)

    return Detector(features,
                    extras,
                    predictor,
                    heads,
                    source_layers,
                    num_classes=num_classes,
                    priors=priors)

def get_extras(source_out_channels,
               use_depthwise=False,
               layers=(),
               activation={'name': 'ReLU', 'args': {'inplace': True}},
               initializer={'name': 'xavier_normal_'},
               batch_norm={}):
    extras = nn.ModuleList()
    in_channels = source_out_channels[-1]
    extra_layers = layers

    for type_, out_channels in extra_layers:
        layers = []

        if type_ == 'm':
            out_channels = in_channels
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif type_ == 's':
            layers.append(conv.Conv2dBn(in_channels, out_channels // 2, kernel_size=1, bias=False,
                                        activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
            in_channels = out_channels // 2
            if use_depthwise:
                layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                     bias=False, activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
            else:
                layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
                                            activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
        elif type_ == '':
            layers.append(conv.Conv2dBn(in_channels, out_channels // 2, kernel_size=1, bias=False,
                                        activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
            in_channels = out_channels // 2
            if use_depthwise:
                layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                                     activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
            else:
                layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                            activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
        else:
            raise ValueError(f'Unknown layer type: {type_}')

        source_out_channels.append(out_channels)
        extras.append(nn.Sequential(*layers))
        in_channels = out_channels

    initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer.get('args', {}))

    def _init_extras(layer):
        if isinstance(layer, nn.Conv2d):
            initializer_(layer.weight)
            layer.bias is not None and nn.init.zeros_(layer.bias)

    extras.apply(_init_extras)

    return extras

def get_predictor(source_out_channels,
                  num_boxes,
                  num_classes,
                  use_depthwise=False,
                  num_layers=0,
                  num_channels=256,
                  kernel_size=3,
                  batch_norm={},
                  activation={'name': 'ReLU', 'args': {'inplace': True}},
                  initializer={'name': 'normal_', 'args': {'mean': 0, 'std': 0.01}},
                  class_head_bias_init=0):
    if num_layers > 0:
        assert len(set(source_out_channels)) == 1

    predictor = nn.ModuleDict()
    norms = nn.ModuleDict()
    for head in ['class', 'loc']:
        in_channels = source_out_channels[0]
        layers = nn.ModuleList()
        norms[head] = nn.ModuleList()

        for _ in range(num_layers):
            if use_depthwise:
                layers.append(conv.DepthwiseConv2dBn(in_channels, num_channels, kernel_size=kernel_size, padding=1,
                                                     bias=True, activation_params=None, use_bn=False))
            else:
                layers.append(conv.Conv2dBn(in_channels, num_channels, kernel_size=kernel_size, padding=1, bias=True,
                                            activation_params=None, use_bn=False))
            layer_norms = nn.ModuleList()
            for _ in source_out_channels:
                layer_norms.append(nn.BatchNorm2d(num_channels, **batch_norm))
            norms[head].append(layer_norms)

            in_channels = num_channels
        predictor[head] = layers

    activation_ = getattr(nn, activation['name'])(**activation.get('args', {}))

    if num_layers > 0:
        out_channels = [num_channels] * len(source_out_channels)
    else:
        out_channels = source_out_channels

    initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer.get('args', {}))

    def _init_predictor(layer):
        if isinstance(layer, nn.Conv2d):
            initializer_(layer.weight)
            nn.init.zeros_(layer.bias)

    predictor.apply(_init_predictor)

    def _init_class_head(layer):
        initializer_(layer.weight)
        nn.init.constant_(layer.bias, class_head_bias_init)

    heads = nn.ModuleList()
    for in_channels, num_boxes in zip(out_channels, num_boxes):
        class_head = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=3, padding=1, bias=True)
        loc_head = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, padding=1, bias=True)

        class_head.apply(_init_class_head)
        loc_head.apply(_init_predictor)

        heads.append(nn.ModuleDict({'class': class_head, 'loc': loc_head}))

    return (predictor, activation_, norms), heads
