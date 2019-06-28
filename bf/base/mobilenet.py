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

class _depthwise_conv_bn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 bias=False,
                 batch_norm_params={}):
        super(_depthwise_conv_bn, self).__init__()

        self.pad = nn.ZeroPad2d((0, 1, 0, 1) if stride == 2 else kernel_size // 2)

        self.depthwise_conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        groups=in_channels,
                                        bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(in_channels, **batch_norm_params)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        self.pointwise_conv = nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        bias=bias)
        self.pointwise_bn = nn.BatchNorm2d(out_channels, **batch_norm_params)
        self.pointwise_relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_relu(x)
        return x

class MobileNet(nn.Module):
    def __init__(self,
                 input_shape=(3, 224, 224),
                 depth_multiplier=1.0,
                 min_depth=4,
                 classes=1000,
                 include_top=False,
                 batch_norm={},
                 init_weights=True):
        super(MobileNet, self).__init__()

        self.input_shape = input_shape
        self.depth_multiplier = depth_multiplier
        self.classes = classes
        self.include_top = include_top
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        conv_bn = functools.partial(_conv_bn,
                                    kernel_size=3,
                                    bias=False,
                                    batch_norm_params=batch_norm)
        depthwise_conv_bn = functools.partial(_depthwise_conv_bn,
                                              kernel_size=3,
                                              bias=False,
                                              batch_norm_params=batch_norm)

        self.features = nn.Sequential(
            conv_bn(input_shape[0], depth(32), stride=2),           # 0
            depthwise_conv_bn(depth(32), depth(64), stride=1),      # 1
            depthwise_conv_bn(depth(64), depth(128), stride=2),     # 2
            depthwise_conv_bn(depth(128), depth(128), stride=1),    # 3
            depthwise_conv_bn(depth(128), depth(256), stride=2),    # 4
            depthwise_conv_bn(depth(256), depth(256), stride=1),    # 5
            depthwise_conv_bn(depth(256), depth(512), stride=2),    # 6
            depthwise_conv_bn(depth(512), depth(512), stride=1),    # 7
            depthwise_conv_bn(depth(512), depth(512), stride=1),    # 8
            depthwise_conv_bn(depth(512), depth(512), stride=1),    # 9
            depthwise_conv_bn(depth(512), depth(512), stride=1),    # 10
            depthwise_conv_bn(depth(512), depth(512), stride=1),    # 11
            depthwise_conv_bn(depth(512), depth(1024), stride=2),   # 12
            depthwise_conv_bn(depth(1024), depth(1024), stride=1),  # 13
        )

        if self.include_top:
            self.logits = nn.Conv2d(depth(1024), classes, kernel_size=1)

        if init_weights:
            self.init()

    def forward(self, x):
        x = self.features(x)

        if not self.include_top:
            return x

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.logits(x)
        x = x.view(x.size(0), -1)
        return F.softmax(x, dim=1)

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
        converter = from_keras().mobilenet(
            input_shape=self.input_shape,
            classes=self.classes,
            include_top=self.include_top,
            depth_multiplier=self.depth_multiplier
        )
        state_dict = self.state_dict()
        self.load_state_dict(converter.to(state_dict))
