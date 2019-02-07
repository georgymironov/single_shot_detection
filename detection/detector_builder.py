import functools
import logging

import torch
import torch.nn as nn

from bf.modules import conv
from bf.modules.features import FeaturePyramid, Features
from detection.detector import Detector
import detection.ssd


def build(base,
          anchor_generator_params,
          num_classes,
          source_layers,
          last_feature_layer=None,
          fpn_layers=None,
          fpn_channels=256,
          depth_multiplier=1.0,
          use_depthwise=False,
          extras={},
          predictor={}):

    extra_layers = extras.get('layers', [])

    if fpn_layers is not None:
        assert fpn_layers >= len(source_layers)
        features = FeaturePyramid(base,
                                  source_layers,
                                  fpn_layers,
                                  fpn_channels,
                                  last_feature_layer=last_feature_layer)
        num_scales = fpn_layers + len(extra_layers)
    else:
        features = Features(base,
                            source_layers,
                            last_feature_layer=last_feature_layer)
        num_scales = len(source_layers) + len(extra_layers)

    source_out_channels = features.get_out_channels()

    priors = get_priors(num_scales, **anchor_generator_params)
    num_boxes = [x.num_boxes for x in priors]

    extras = get_extras(source_out_channels, use_depthwise, depth_multiplier, **extras)
    predictor, heads = get_predictor(source_out_channels, num_boxes, num_classes, use_depthwise, **predictor)

    return Detector(num_classes,
                    features,
                    extras,
                    predictor,
                    heads,
                    priors,
                    source_layers)

def get_extras(source_out_channels,
               use_depthwise=False,
               depth_multiplier=1.0,
               layers=(),
               activation={'name': 'ReLU', 'args': {'inplace': True}},
               batch_norm={}):
    extras = nn.ModuleList()
    in_channels = source_out_channels[-1]
    extra_layers = layers

    for type_, depth in extra_layers:
        if depth is None:
            continue

        out_channels = int(depth * depth_multiplier)
        layers = []

        layers.append(conv.Conv2dBn(in_channels, out_channels // 2, kernel_size=1, bias=False,
                                    activation_params=activation, use_bn=True, batch_norm_params=batch_norm))

        in_channels = out_channels // 2

        if type_ == 's':
            if use_depthwise:
                layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                     bias=False, activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
            else:
                layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
                                            activation_params=activation, use_bn=True, batch_norm_params=batch_norm))

        elif type_ == '':
            if use_depthwise:
                layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                                     activation_params=activation, use_bn=True, batch_norm_params=batch_norm))
            else:
                layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                            activation_params=activation, use_bn=True, batch_norm_params=batch_norm))

        source_out_channels.append(out_channels)
        extras.append(nn.Sequential(*layers))
        in_channels = out_channels

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
    for head in ['class', 'loc']:
        in_channels = source_out_channels[0]
        layers = []
        for _ in range(num_layers):
            if use_depthwise:
                layers.append(conv.DepthwiseConv2dBn(in_channels, num_channels, kernel_size=kernel_size, padding=1,
                                                     bias=True, activation_params=activation, use_bn=False))
            else:
                layers.append(conv.Conv2dBn(in_channels, num_channels, kernel_size=kernel_size, padding=1, bias=True,
                                            activation_params=activation, use_bn=False))
            in_channels = num_channels

        predictor[head] = nn.Sequential(*layers)

    if num_layers > 0:
        out_channels = [num_channels] * len(source_out_channels)
    else:
        out_channels = source_out_channels

    initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer['args'])

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
        class_head = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=3, padding=1)
        loc_head = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, padding=1)

        class_head.apply(_init_class_head)
        loc_head.apply(_init_predictor)

        heads.append(nn.ModuleDict({'class': class_head, 'loc': loc_head}))

    return predictor, heads

def get_priors(num_scales,
               type,
               sizes=None,
               min_scale=None,
               max_scale=None,
               aspect_ratios=[[1.0, 2.0]] + [[1.0, 2.0, 3.0]] * 3 + [[1.0, 2.0]] * 2,
               steps=None,
               offsets=[0.5, 0.5],
               num_branches=None):
    assert sizes is not None or (min_scale is not None and max_scale is not None)

    if steps is None:
        steps = [None] * num_scales
    else:
        assert len(steps) == num_scales

    if num_branches is None:
        num_branches = [1] * num_scales
    else:
        assert len(num_branches) == num_scales

    if min_scale is not None and max_scale is not None:
        scales = torch.linspace(min_scale, max_scale, num_scales + 1)
        logging.info(f'Detector (Scales: {scales[:-1]})')
    else:
        scales = None

    assert len(aspect_ratios) == num_scales

    AnchorGenerator = getattr(detection, type).AnchorGenerator

    priors = []
    for i, (ratios, step, num_branches) in enumerate(zip(aspect_ratios, steps, num_branches)):
        if scales is not None:
            kwargs = {
                'min_scale': scales[i],
                'max_scale': scales[i + 1]
            }
        else:
            kwargs = {
                'min_size': sizes[i],
                'max_size': sizes[i + 1]
            }
        priors.append(AnchorGenerator(ratios, step=step, num_branches=num_branches, **kwargs))
    return priors
