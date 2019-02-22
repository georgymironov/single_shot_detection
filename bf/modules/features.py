import functools
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from bf.modules import conv
from bf.utils.torch_utils import get_multiple_outputs


class Features(nn.Module):
    def __init__(self,
                 base,
                 out_layers,
                 last_feature_layer=None):
        super(Features, self).__init__()

        assert isinstance(base.features, nn.Sequential)

        feature_layers = list(base.features.children())
        if last_feature_layer is not None:
            feature_layers = feature_layers[:(last_feature_layer + 1)]

        self.base = nn.Sequential(*feature_layers)
        self.out_layers = out_layers

        self.num_outputs = len(out_layers)

    def forward(self, x):
        with torch.jit.scope('Sequential[base]'):
            sources, x = get_multiple_outputs(self.base, x, self.out_layers)
        return sources, x

    def get_out_channels(self):
        dummy = torch.ones((1, 3, 300, 300), dtype=torch.float)
        sources, _ = get_multiple_outputs(self.base, dummy, self.out_layers)
        return [x.size(1) for x in sources]

class FeaturePyramid(Features):
    def __init__(self,
                 base,
                 out_layers,
                 pyramid_layers,
                 pyramid_channels,
                 interpolation_mode='nearest',
                 use_depthwise=False,
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 initializer={'name': 'xavier_normal_'},
                 **kwargs):
        super(FeaturePyramid, self).__init__(base, out_layers, **kwargs)

        assert pyramid_layers >= len(out_layers)

        self.pyramid_layers = pyramid_layers
        self.pyramid_channels = pyramid_channels
        self.interpolation_mode = interpolation_mode
        self.use_depthwise = use_depthwise

        self.num_outputs = pyramid_layers

        self.pyramid_lateral = nn.ModuleList()
        self.pyramid_output = nn.ModuleList()

        base_out_channels = super(FeaturePyramid, self).get_out_channels()

        conv_op = conv.DepthwiseConv2dBn if self.use_depthwise else conv.Conv2dBn

        for in_channels in base_out_channels:
            self.pyramid_lateral.append(nn.Conv2d(in_channels, pyramid_channels, kernel_size=1))
            self.pyramid_output.append(conv_op(pyramid_channels,
                                               pyramid_channels,
                                               kernel_size=3,
                                               padding=1,
                                               activation_params=activation))

        for i in range(pyramid_layers - len(base_out_channels)):
            self.pyramid_output.append(conv_op(pyramid_channels,
                                               pyramid_channels,
                                               kernel_size=3,
                                               padding=1,
                                               stride=2,
                                               activation_params=activation))

        initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer.get('args', {}))

        def _init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                initializer_(layer.weight)
                layer.bias is not None and nn.init.zeros_(layer.bias)

        self.pyramid_lateral.apply(_init_layer)
        self.pyramid_output.apply(_init_layer)

    def forward(self, feature):
        sources, _ = super(FeaturePyramid, self).forward(feature)

        features = [lateral(source) for source, lateral in zip(sources, self.pyramid_lateral)]

        for i in reversed(range(len(features) - 1)):
            features[i] += F.interpolate(features[i + 1], size=features[i].size()[2:], mode='nearest')

        outputs = []
        for output_layer, feature in itertools.zip_longest(self.pyramid_output, features):
            if feature is not None:
                outputs.append(output_layer(feature))
            else:
                outputs.append(output_layer(outputs[-1]))

        return outputs, outputs[-1]

    def get_out_channels(self):
        return [self.pyramid_channels] * self.pyramid_layers
