import functools

import torch.nn as nn
import torch.nn.functional as F


class _conv_bn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False,
                 batch_norm_params={}):
        super(_conv_bn, self).__init__()

        self.pad = nn.ZeroPad2d((0, 1, 0, 1) if stride == 2 else kernel_size // 2)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, **batch_norm_params)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expantion_ratio,
                 bias=False,
                 batch_norm_params={}):
        super(_inverted_residual_bottleneck, self).__init__()

        inner_channels = in_channels * expantion_ratio
        self.residual = in_channels == out_channels and stride == 1

        if inner_channels > in_channels:
            self.expand_conv = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=bias)
            self.expand_bn = nn.BatchNorm2d(in_channels * expantion_ratio, **batch_norm_params)
            self.expand_relu = nn.ReLU6(inplace=True)

        self.pad = nn.ZeroPad2d((0, 1, 0, 1) if stride == 2 else 1)
        self.depthwise_conv = nn.Conv2d(inner_channels, inner_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        groups=inner_channels,
                                        bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(inner_channels, **batch_norm_params)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        self.project_conv = nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=bias)
        self.project_bn = nn.BatchNorm2d(out_channels, **batch_norm_params)

    def forward(self, input_):
        if 'expand_conv' in self._modules:
            x = self.expand_conv(input_)
            x = self.expand_bn(x)
            x = self.expand_relu(x)
        else:
            x = input_

        x = self.pad(x)
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)
        x = self.project_conv(x)
        x = self.project_bn(x)

        return input_ + x if self.residual else x


class MobileNetV2(nn.Module):
    def __init__(self,
                 input_shape=(3, 224, 224),
                 depth_multiplier=1.0,
                 min_depth=4,
                 classes=1000,
                 include_top=False,
                 batch_norm={},
                 init_weights=True):
        super(MobileNetV2, self).__init__()

        self.input_shape = input_shape
        self.depth_multiplier = depth_multiplier
        self.classes = classes
        self.include_top = include_top
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        conv_bn = functools.partial(_conv_bn,
                                    bias=False,
                                    batch_norm_params=batch_norm)
        inverted_residual_bottleneck = functools.partial(_inverted_residual_bottleneck,
                                                         bias=False,
                                                         batch_norm_params=batch_norm)

        self.features = nn.Sequential(
            conv_bn(input_shape[0], depth(32), kernel_size=3, stride=2),                        # 0
            inverted_residual_bottleneck(depth(32), depth(16), stride=1, expantion_ratio=1),    # 1
            inverted_residual_bottleneck(depth(16), depth(24), stride=2, expantion_ratio=6),    # 2
            inverted_residual_bottleneck(depth(24), depth(24), stride=1, expantion_ratio=6),    # 3
            inverted_residual_bottleneck(depth(24), depth(32), stride=2, expantion_ratio=6),    # 4
            inverted_residual_bottleneck(depth(32), depth(32), stride=1, expantion_ratio=6),    # 5
            inverted_residual_bottleneck(depth(32), depth(32), stride=1, expantion_ratio=6),    # 6
            inverted_residual_bottleneck(depth(32), depth(64), stride=2, expantion_ratio=6),    # 7
            inverted_residual_bottleneck(depth(64), depth(64), stride=1, expantion_ratio=6),    # 8
            inverted_residual_bottleneck(depth(64), depth(64), stride=1, expantion_ratio=6),    # 9
            inverted_residual_bottleneck(depth(64), depth(64), stride=1, expantion_ratio=6),    # 10
            inverted_residual_bottleneck(depth(64), depth(96), stride=1, expantion_ratio=6),    # 11
            inverted_residual_bottleneck(depth(96), depth(96), stride=1, expantion_ratio=6),    # 12
            inverted_residual_bottleneck(depth(96), depth(96), stride=1, expantion_ratio=6),    # 13
            inverted_residual_bottleneck(depth(96), depth(160), stride=2, expantion_ratio=6),   # 14
            inverted_residual_bottleneck(depth(160), depth(160), stride=1, expantion_ratio=6),  # 15
            inverted_residual_bottleneck(depth(160), depth(160), stride=1, expantion_ratio=6),  # 16
            inverted_residual_bottleneck(depth(160), depth(320), stride=1, expantion_ratio=6),  # 17
            conv_bn(depth(320), depth(1280), kernel_size=1)                                     # 18
        )

        if self.include_top:
            self.logits = nn.Linear(depth(1280), classes)

        if init_weights:
            self.init()

    def forward(self, x):
        x = self.features(x)

        if not self.include_top:
            return x

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.logits(x)
        return F.softmax(x)

    @staticmethod
    def init_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def init(self):
        self.apply(self.init_layer)

    def init_from_keras(self):
        from bf.utils.convert_weights import from_keras
        converter = from_keras().mobilenet_v2(
            input_shape=self.input_shape,
            classes=self.classes,
            include_top=self.include_top,
            depth_multiplier=self.depth_multiplier
        )
        state_dict = self.state_dict()
        self.load_state_dict(converter.to(state_dict))
