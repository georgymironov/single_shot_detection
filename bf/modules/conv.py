import torch.nn as nn


class Conv2dBn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 activation_params={'name': 'ReLU', 'args': {'inplace': True}},
                 batch_norm_params={}):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels, **batch_norm_params)
        if activation_params is not None:
            self.activation = getattr(nn, activation_params['name'])(**activation_params['args'])

    def forward(self, x):
        x = self.conv(x)
        if 'bn' in self._modules:
            x = self.bn(x)
        if 'activation' in self._modules:
            x = self.activation(x)
        return x


class DepthwiseConv2dBn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 use_bn=True,
                 activation_params={'name': 'ReLU', 'args': {'inplace': True}},
                 batch_norm_params={}):
        super(DepthwiseConv2dBn, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=in_channels,
                                        bias=bias)
        if use_bn:
            self.depthwise_bn = nn.BatchNorm2d(in_channels, **batch_norm_params)
        if activation_params is not None:
            self.depthwise_activation = getattr(nn, activation_params['name'])(**activation_params['args'])

        self.pointwise_conv = nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        bias=bias)
        if use_bn:
            self.pointwise_bn = nn.BatchNorm2d(out_channels, **batch_norm_params)
        if activation_params is not None:
            self.pointwise_activation = getattr(nn, activation_params['name'])(**activation_params['args'])

    def forward(self, x):
        x = self.depthwise_conv(x)
        if 'depthwise_bn' in self._modules:
            x = self.depthwise_bn(x)
        if 'depthwise_activation' in self._modules:
            x = self.depthwise_activation(x)
        x = self.pointwise_conv(x)
        if 'pointwise_bn' in self._modules:
            x = self.pointwise_bn(x)
        if 'pointwise_activation' in self._modules:
            x = self.pointwise_activation(x)

        return x
