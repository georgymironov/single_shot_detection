import functools

import torch.nn as nn

from bf.modules import conv


class SharedConvPredictor(nn.Module):
    def __init__(self,
                 source_out_channels,
                 num_boxes,
                 num_classes,
                 use_depthwise,
                 num_layers=0,
                 num_channels=256,
                 kernel_size=3,
                 batch_norm={},
                 activation={'name': 'ReLU', 'args': {'inplace': True}},
                 initializer={'name': 'normal_', 'args': {'mean': 0, 'std': 0.01}}):
        super(SharedConvPredictor, self).__init__()

        if num_layers > 0:
            assert len(set(source_out_channels)) == 1

        self.convs = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for head in ['score', 'loc']:
            in_channels = source_out_channels[0]
            layers = nn.ModuleList()
            self.norms[head] = nn.ModuleList()

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
                self.norms[head].append(layer_norms)

                in_channels = num_channels
            self.convs[head] = layers

        self.activation = getattr(nn, activation['name'])(**activation.get('args', {}))

        self.out_channels = [num_channels] * len(source_out_channels)

        initializer_ = functools.partial(getattr(nn.init, initializer['name']), **initializer.get('args', {}))

        def _init_predictor(layer):
            if isinstance(layer, nn.Conv2d):
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.convs.apply(_init_predictor)

    def forward(self, sources):
        score_sources = loc_sources = sources

        for score_conv, loc_conv, score_norm, loc_norm in zip(self.convs['score'],
                                                              self.convs['loc'],
                                                              self.norms['score'],
                                                              self.norms['loc']):
            score_sources = map(score_conv, score_sources)
            loc_sources = map(loc_conv, loc_sources)

            score_sources = map(self.activation, score_sources)
            loc_sources = map(self.activation, loc_sources)

            score_sources = [norm(x) for norm, x in zip(score_norm, score_sources)]
            loc_sources = [norm(x) for norm, x in zip(loc_norm, loc_sources)]

        return score_sources, loc_sources
