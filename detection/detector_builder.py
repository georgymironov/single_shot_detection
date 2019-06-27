import functools

import torch.nn as nn

from bf.modules import conv
from bf.modules import features as _features
from detection import anchor_generators as _anchor_generators
from detection.detector import Detector
from detection.modules import predictors


def build(base,
          anchor_generator_params,
          num_classes,
          features,
          use_depthwise=False,
          extras={},
          predictor={},
          heads={}):

    extra_layers = extras.get('layers', [])

    Features = getattr(_features, features['name'])
    features = Features(base, use_depthwise=use_depthwise, **features)

    num_scales = features.num_outputs + len(extra_layers)
    source_out_channels = features.get_out_channels()

    anchor_generator_builder = getattr(_anchor_generators, anchor_generator_params['type']).build_anchor_generators
    anchor_generators = anchor_generator_builder(**anchor_generator_params)

    assert num_scales == len(anchor_generators)
    num_boxes = [x.num_boxes for x in anchor_generators]

    extras = get_extras(source_out_channels, use_depthwise=use_depthwise, **extras)

    predictor = get_predictor(source_out_channels,
                              num_boxes,
                              num_classes,
                              use_depthwise,
                              predictor_args=predictor)

    out_channels = predictor.out_channels if predictor else source_out_channels

    heads = get_heads(out_channels,
                      num_boxes,
                      num_classes,
                      **heads)

    return Detector(features,
                    extras,
                    predictor,
                    heads,
                    num_classes,
                    anchor_generators=anchor_generators)

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

def get_heads(out_channels,
              num_boxes,
              num_classes,
              initializer={'name': 'normal_', 'args': {'mean': 0, 'std': 0.01}},
              score_head_bias_init=0.0):
    initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer.get('args', {}))

    def _init_head(layer):
        initializer_(layer.weight)
        nn.init.zeros_(layer.bias)

    def _init_score_head(layer):
        initializer_(layer.weight)
        nn.init.constant_(layer.bias, score_head_bias_init)

    heads = nn.ModuleList()

    for in_channels, num_boxes in zip(out_channels, num_boxes):
        score_head = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=3, padding=1, bias=True)
        loc_head = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, padding=1, bias=True)

        score_head.apply(_init_score_head)
        loc_head.apply(_init_head)

        heads.append(nn.ModuleDict({'score': score_head, 'loc': loc_head}))

    return heads

def get_predictor(source_out_channels,
                  num_boxes,
                  num_classes,
                  use_depthwise,
                  predictor_args):

    if not predictor_args:
        return None

    predictor = predictors.SharedConvPredictor(source_out_channels, num_boxes, num_classes, use_depthwise, **predictor_args)

    return predictor
