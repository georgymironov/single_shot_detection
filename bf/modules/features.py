import functools
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from bf.modules import conv
from bf.utils.misc_utils import update_existing
from bf.utils.torch_utils import get_multiple_outputs


def _init_layer(layer, initializer_):
    if isinstance(layer, nn.Conv2d):
        initializer_(layer.weight)
        layer.bias is not None and nn.init.zeros_(layer.bias)

class Features(nn.Module):
    def __init__(self,
                 base,
                 out_layers,
                 last_feature_layer=None,
                 initializer={'name': 'xavier_normal_'}):
        super(Features, self).__init__()

        assert isinstance(base.features, nn.Sequential)

        feature_layers = list(base.features.children())
        if last_feature_layer is not None:
            feature_layers = feature_layers[:(last_feature_layer + 1)]

        self.base = nn.Sequential(*feature_layers)
        self.out_layers = out_layers
        self.num_outputs = len(out_layers)

        initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer.get('args', {}))
        self.init_layer = functools.partial(_init_layer, initializer_=initializer_)

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

        if self.use_depthwise:
            conv_op = functools.partial(conv.Conv2dBn, groups=pyramid_channels)
        else:
            conv_op = conv.Conv2dBn

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

        self.pyramid_lateral.apply(self.init_layer)
        self.pyramid_output.apply(self.init_layer)

    def forward(self, x):
        sources, _ = super(FeaturePyramid, self).forward(x)
        features = [lateral(source) for source, lateral in zip(sources, self.pyramid_lateral)]

        for i in reversed(range(len(features) - 1)):
            features[i] += F.interpolate(features[i + 1], size=features[i].size()[2:], mode=self.interpolation_mode)

        outputs = []
        for output_layer, feature in itertools.zip_longest(self.pyramid_output, features):
            if feature is not None:
                outputs.append(output_layer(feature))
            else:
                outputs.append(output_layer(outputs[-1]))

        return outputs, outputs[-1]

    def get_out_channels(self):
        return [self.pyramid_channels] * self.pyramid_layers

# ref: https://arxiv.org/pdf/1807.11013.pdf
class DepthwiseFeaturePyramid(Features):
    def __init__(self,
                 base,
                 out_layers,
                 pyramid_layers,
                 pyramid_channels,
                 interpolation_mode='nearest',
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 initializer={'name': 'xavier_normal_'},
                 **kwargs):
        super(DepthwiseFeaturePyramid, self).__init__(base, out_layers, **kwargs)

        self.pyramid_layers = pyramid_layers
        self.pyramid_channels = pyramid_channels
        self.interpolation_mode = interpolation_mode
        self.pyramid_lateral = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.up_conv = nn.ModuleList()

        self.num_outputs = pyramid_layers

        base_out_channels = super(DepthwiseFeaturePyramid, self).get_out_channels()

        for in_channels in base_out_channels:
            self.pyramid_lateral.append(nn.Conv2d(in_channels, pyramid_channels, kernel_size=1))

        for _ in range(pyramid_layers - len(out_layers)):
            paths = nn.ModuleList()
            paths.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                conv.Conv2dBn(pyramid_channels,
                              pyramid_channels // 2,
                              kernel_size=1,
                              activation_params=activation)))
            paths.append(conv.DepthwiseConv2dBn(pyramid_channels,
                                                pyramid_channels // 2,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                activation_params=activation))
            self.downsample.append(paths)

        for _ in range(pyramid_layers - 1):
            self.up_conv.append(conv.Conv2dBn(pyramid_channels,
                                              pyramid_channels,
                                              kernel_size=3,
                                              padding=1,
                                              groups=pyramid_channels,
                                              activation_params=activation))

        self.pyramid_lateral.apply(self.init_layer)
        self.downsample.apply(self.init_layer)
        self.up_conv.apply(self.init_layer)

    def forward(self, x):
        sources, _ = super(DepthwiseFeaturePyramid, self).forward(x)
        features = [lateral(source) for source, lateral in zip(sources, self.pyramid_lateral)]

        for down in self.downsample:
            padding = [0, 0, 0, 0]
            if (features[-1].shape[3] > 2):
                padding[0:2] = [0, 1]
            if (features[-1].shape[2] > 2):
                padding[2:4] = [0, 1]
            first = down[0](F.pad(features[-1], padding))
            second = down[1](features[-1])
            features.append(torch.cat([first, second], dim=1))

        output = [features[-1]]
        for i in reversed(range(0, len(features) - 1)):
            up = F.interpolate(output[-1], size=features[i].size()[2:], mode=self.interpolation_mode)
            output.append(self.up_conv[i](up) + features[i])

        output = list(reversed(output))

        return output, output[-1]

    def get_out_channels(self):
        return [self.pyramid_channels] * self.pyramid_layers

# ref: https://qijiezhao.github.io/imgs/m2det.pdf
class ThinnedUshapeModule(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 num_scales,
                 interpolation_mode='nearest',
                 use_depthwise=False,
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 initializer={'name': 'xavier_normal_'}):
        super(ThinnedUshapeModule, self).__init__()

        self.interpolation_mode = interpolation_mode

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()

        conv_op = conv.DepthwiseConv2dBn if use_depthwise else conv.Conv2dBn

        for i in range(num_scales):
            if i > 0:
                self.down_layers.append(conv_op(in_channels if i == 1 else inner_channels,
                                                inner_channels,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                activation_params=activation))
                self.up_layers.append(conv_op(inner_channels,
                                              in_channels if i == 1 else inner_channels,
                                              kernel_size=1,
                                              activation_params=activation))

            self.smooth_layers.append(conv_op(in_channels if i == 0 else inner_channels,
                                              out_channels,
                                              kernel_size=1,
                                              activation_params=activation))

    def forward(self, x):
        down_path = [x]

        for layer in self.down_layers:
            x = layer(x)
            down_path.append(x)

        up_path = [x]

        for down_x, layer in zip(reversed(down_path[:-1]), reversed(self.up_layers)):
            x = layer(x)
            x = F.interpolate(x, size=down_x.size()[2:], mode=self.interpolation_mode)
            x += down_x
            up_path.append(x)

        out = [layer(x) for layer, x in zip(reversed(self.smooth_layers), up_path)]

        return out

# ref: https://qijiezhao.github.io/imgs/m2det.pdf
class ScalewiseFeatureAggregationModule(nn.Module):
    def __init__(self,
                 num_channels,
                 num_scales,
                 reduction_ratio=16):
        super(ScalewiseFeatureAggregationModule, self).__init__()

        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()

        for _ in range(num_scales):
            self.fc1.append(nn.Conv2d(num_channels, num_channels // reduction_ratio, kernel_size=1))
            self.fc2.append(nn.Conv2d(num_channels // reduction_ratio, num_channels, kernel_size=1))

    def forward(self, features):
        assert len(features) == len(self.fc1)

        result = []

        for feature, fc1, fc2 in zip(features, self.fc1, self.fc2):
            x = F.adaptive_avg_pool2d(feature, 1)
            x = fc1(x)
            x = F.relu(x)
            x = fc2(x)
            x = torch.sigmoid(x)
            result.append(feature * x)

        return result

# ref: https://qijiezhao.github.io/imgs/m2det.pdf
class MultilevelFeaturePyramid(Features):
    def __init__(self,
                 base,
                 out_layers,
                 num_scales,
                 num_tums,
                 base_reduced_channels=[256, 512],
                 reduced_channels=128,
                 interpolation_mode='nearest',
                 use_depthwise=False,
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 initializer={'name': 'xavier_normal_'},
                 tum={'inner_channels': 256, 'out_channels': 128},
                 sfam={'reduction_ratio': 16},
                 **kwargs):
        super(MultilevelFeaturePyramid, self).__init__(base, out_layers, **kwargs)

        assert len(out_layers) == len(base_reduced_channels)
        assert num_tums > 0

        self.num_outputs = num_scales
        self.num_tums = num_tums
        self.interpolation_mode = interpolation_mode

        self.base_reducers = nn.ModuleList()
        base_out_channels = super(MultilevelFeaturePyramid, self).get_out_channels()

        for in_channels, out_channels in zip(base_out_channels, base_reduced_channels):
            self.base_reducers.append(conv.Conv2dBn(in_channels, out_channels, kernel_size=1, activation_params=activation))

        tum.update({
            'num_scales': num_scales
        })
        update_existing(tum, {
            'interpolation_mode': interpolation_mode,
            'use_depthwise': use_depthwise,
            'activation': activation
        })
        self.tum_out_channels = tum['out_channels']

        self.tums = nn.ModuleList()
        self.reducers = nn.ModuleList()

        self.tums.append(ThinnedUshapeModule(in_channels=sum(base_reduced_channels), **tum))

        for i in range(1, num_tums):
            self.tums.append(ThinnedUshapeModule(in_channels=reduced_channels + self.tum_out_channels, **tum))
            self.reducers.append(conv.Conv2dBn(sum(base_reduced_channels),
                                               reduced_channels,
                                               kernel_size=1,
                                               activation_params=activation))

        sfam.update({
            'num_channels': self.tum_out_channels * self.num_tums,
            'num_scales': num_scales
        })

        self.sfam = ScalewiseFeatureAggregationModule(**sfam)

        self.base_reducers.apply(self.init_layer)
        self.tums.apply(self.init_layer)
        self.reducers.apply(self.init_layer)
        self.sfam.apply(self.init_layer)

    def forward(self, x):
        sources, _ = super(MultilevelFeaturePyramid, self).forward(x)
        base_reduced = [reducer(source) for reducer, source in zip(self.base_reducers, sources)]

        upscaled = [base_reduced[0]]
        for features in base_reduced[1:]:
            upscaled.append(F.interpolate(features, size=base_reduced[0].size()[2:], mode=self.interpolation_mode))

        base_features = torch.cat(upscaled, dim=1)
        features = self.tums[0](base_features)
        features = [[x] for x in features]

        for tum, reducer in zip(self.tums[1:], self.reducers):
            x = features[-1][-1]
            reduced = reducer(base_features)
            x = torch.cat([x, reduced], dim=1)

            for i, feature in enumerate(tum(x)):
                features[i].append(feature)

        features = [torch.cat(x, dim=1) for x in reversed(features)]
        features = self.sfam(features)

        return features, features[-1]

    def get_out_channels(self):
        return [self.tum_out_channels * self.num_tums] * self.num_outputs
