import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        with torch.jit.scope('Sequential[base]'):
            sources, x = get_multiple_outputs(self.base, x, self.out_layers)
        return sources, x

    def get_out_channels(self):
        dummy = torch.ones((1, 3, 300, 300), dtype=torch.float)
        sources, _ = get_multiple_outputs(self.base, dummy, self.out_layers)
        return [x.size(1) for x in sources]

class FeaturePyramid(Features):
    def __init__(self, base, out_layers, pyramid_layers, pyramid_channels, **kwargs):
        super(FeaturePyramid, self).__init__(base, out_layers, **kwargs)

        self.pyramid_layers = pyramid_layers
        self.pyramid_channels = pyramid_channels

        self.pyramid_lateral = nn.ModuleList()
        self.pyramid_output = nn.ModuleList()

        base_out_channels = super(FeaturePyramid, self).get_out_channels()

        for in_channels in base_out_channels:
            self.pyramid_lateral.append(nn.Conv2d(in_channels, pyramid_channels, kernel_size=1))
            self.pyramid_output.append(nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1))

        for i in range(pyramid_layers - len(base_out_channels)):
            if i == 0:
                self.pyramid_output.append(
                    nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, stride=2)
                )
            else:
                self.pyramid_output.append(nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, stride=2)
                ))

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
