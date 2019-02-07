import logging

import torch
import torch.nn as nn

from bf.modules import conv, features
from detection.detector import Detector
import detection.ssd


class DetectorBuilder(object):
    def __init__(self,
                 base,
                 anchor_generator_params,
                 num_classes,
                 source_layers,
                 extra_layers=(),
                 last_feature_layer=None,
                 fpn_layers=None,
                 fpn_channels=256,
                 depth_multiplier=1.0,
                 use_depthwise=False,
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 batch_norm={}):

        self.num_classes = num_classes
        self.source_layers = source_layers
        self.extra_layers = extra_layers
        self.depth_multiplier = depth_multiplier
        self.use_depthwise = use_depthwise
        self.activation_params = activation
        self.batch_norm_params = batch_norm

        if fpn_layers is not None:
            assert fpn_layers >= len(source_layers)
            self.features = features.FeaturePyramid(base,
                                                    source_layers,
                                                    fpn_layers,
                                                    fpn_channels,
                                                    last_feature_layer=last_feature_layer)
            self.num_scales = fpn_layers + len(extra_layers)
        else:
            self.features = features.Features(base,
                                              source_layers,
                                              last_feature_layer=last_feature_layer)
            self.num_scales = len(source_layers) + len(extra_layers)

        self.source_out_channels = self.features.get_out_channels()

        self.priors = self.get_priors(**anchor_generator_params)
        self.num_boxes = [x.num_boxes for x in self.priors]

    def build(self):
        extras = self.get_extras()
        heads = self.get_heads()

        return Detector(self.num_classes,
                        self.features,
                        extras,
                        heads,
                        self.priors,
                        self.source_layers)

    def get_extras(self):
        extras = nn.ModuleList()
        in_channels = self.source_out_channels[-1]

        for type_, depth in self.extra_layers:
            if depth is None:
                continue

            out_channels = int(depth * self.depth_multiplier)
            layers = []

            layers.append(conv.Conv2dBn(in_channels, out_channels // 2, kernel_size=1, bias=False,
                                        activation_params=self.activation_params,
                                        use_bn=True, batch_norm_params=self.batch_norm_params))

            in_channels = out_channels // 2

            if type_ == 's':
                if self.use_depthwise:
                    layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, stride=2,
                                                         padding=1, bias=False, activation_params=self.activation_params,
                                                         use_bn=True, batch_norm_params=self.batch_norm_params))
                else:
                    layers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=3, stride=2,
                                                padding=1, bias=False, activation_params=self.activation_params,
                                                use_bn=True, batch_norm_params=self.batch_norm_params))

            elif type_ == '':
                if self.use_depthwise:
                    layers.append(conv.DepthwiseConv2dBn(in_channels, out_channels, kernel_size=3, bias=False,
                                                         activation_params=self.activation_params,
                                                         use_bn=True, batch_norm_params=self.batch_norm_params))
                else:
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

    def get_priors(self,
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
            steps = [None] * self.num_scales
        else:
            assert len(steps) == self.num_scales

        if num_branches is None:
            num_branches = [1] * self.num_scales
        else:
            assert len(num_branches) == self.num_scales

        if min_scale is not None and max_scale is not None:
            scales = torch.linspace(min_scale, max_scale, self.num_scales + 1)
            logging.info(f'Detector (Scales: {scales[:-1]})')
        else:
            scales = None

        assert len(aspect_ratios) == self.num_scales

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
