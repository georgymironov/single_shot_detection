from collections import namedtuple
import logging
import os

import torch


LayerMap = namedtuple('LayerMap', ['type', 'src', 'dst'])

class from_keras(object):
    state_dict = None

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras import backend as K

        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))

    def _get_weights(self, name):
        if name not in self.source_layers:
            logging.warning(f'{name} layer is missing in source model')
            return []

        layer = self.model.get_layer(name)
        return zip([x.name for x in layer.weights],
                   layer.get_weights())

    def _set_weight(self, weight, src, dst, permute=None):
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        if permute:
            weight_tensor = weight_tensor.permute(permute)
        assert weight_tensor.size() == self.state_dict[dst].size()
        self.state_dict[dst] = weight_tensor
        logging.debug(f'{src} -> {dst}')

    def _set_conv(self, src, dst):
        weights = self._get_weights(src)
        for name, weight in weights:
            if name.endswith('/kernel:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.weight', permute=(3, 2, 0, 1))
            elif name.endswith('/depthwise_kernel:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.weight', permute=(2, 3, 0, 1))
            elif name.endswith('/bias:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.bias')

    def _set_bn(self, src, dst):
        weights = self._get_weights(src)
        for name, weight in weights:
            if name.endswith('/gamma:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.weight')
            elif name.endswith('/beta:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.bias')
            elif name.endswith('/moving_mean:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.running_mean')
            elif name.endswith('/moving_variance:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.running_var')

    def _set_fc(self, src, dst):
        weights = self._get_weights(src)
        for name, weight in weights:
            if name.endswith('/kernel:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.weight', permute=(1, 0))
            elif name.endswith('/bias:0'):
                self._set_weight(weight, src=name, dst=f'{dst}.bias')

    def mobilenet(self, input_shape, classes, include_top=False, depth_multiplier=1.0):
        from keras.applications.mobilenet import MobileNet

        self.model = MobileNet(
            input_shape=(input_shape[1], input_shape[2], input_shape[0]),
            alpha=depth_multiplier,
            include_top=include_top,
            classes=classes)

        self.source_layers = [x.name for x in self.model.layers]

        self.mapping = [
            LayerMap(type='conv', src='conv1', dst='features.0.conv'),
            LayerMap(type='bn', src='conv1_bn', dst='features.0.bn'),
        ]

        for i in range(1, 14):
            self.mapping.append(LayerMap(type='conv', src=f'conv_dw_{i}', dst=f'features.{i}.depthwise_conv'))
            self.mapping.append(LayerMap(type='bn', src=f'conv_dw_{i}_bn', dst=f'features.{i}.depthwise_bn'))
            self.mapping.append(LayerMap(type='conv', src=f'conv_pw_{i}', dst=f'features.{i}.pointwise_conv'))
            self.mapping.append(LayerMap(type='bn', src=f'conv_pw_{i}_bn', dst=f'features.{i}.pointwise_bn'))

        if include_top:
            self.mapping.append(LayerMap(type='conv', src='conv_preds', dst='logits'))

        return self

    def mobilenet_v2(self, input_shape, classes, include_top=False, depth_multiplier=1.0):
        from keras.applications.mobilenet_v2 import MobileNetV2

        self.model = MobileNetV2(
            input_shape=(input_shape[1], input_shape[2], input_shape[0]),
            alpha=depth_multiplier,
            include_top=include_top,
            classes=classes)

        self.source_layers = [x.name for x in self.model.layers]

        self.mapping = [
            LayerMap(type='conv', src='Conv1', dst='features.0.conv'),
            LayerMap(type='bn', src='bn_Conv1', dst='features.0.bn'),
            LayerMap(type='conv', src='expanded_conv_depthwise', dst='features.1.depthwise_conv'),
            LayerMap(type='bn', src='expanded_conv_depthwise_BN', dst='features.1.depthwise_bn'),
            LayerMap(type='conv', src='expanded_conv_project', dst='features.1.project_conv'),
            LayerMap(type='bn', src='expanded_conv_project_BN', dst='features.1.project_bn'),
        ]

        for i in range(2, 18):
            self.mapping.append(LayerMap(type='conv', src=f'block_{i-1}_expand', dst=f'features.{i}.expand_conv'))
            self.mapping.append(LayerMap(type='bn', src=f'block_{i-1}_expand_BN', dst=f'features.{i}.expand_bn'))
            self.mapping.append(LayerMap(type='conv', src=f'block_{i-1}_depthwise', dst=f'features.{i}.depthwise_conv'))
            self.mapping.append(LayerMap(type='bn', src=f'block_{i-1}_depthwise_BN', dst=f'features.{i}.depthwise_bn'))
            self.mapping.append(LayerMap(type='conv', src=f'block_{i-1}_project', dst=f'features.{i}.project_conv'))
            self.mapping.append(LayerMap(type='bn', src=f'block_{i-1}_project_BN', dst=f'features.{i}.project_bn'))

        self.mapping.append(LayerMap(type='conv', src='Conv_1', dst='features.18.conv'))
        self.mapping.append(LayerMap(type='bn', src='Conv_1_bn', dst='features.18.bn'))

        if include_top:
            self.mapping.append(LayerMap(type='fc', src='Logits', dst='logits'))

        return self

    def to(self, state_dict):
        self.state_dict = state_dict

        for layer in self.mapping:
            if layer.type == 'conv':
                self._set_conv(layer.src, layer.dst)
            if layer.type == 'bn':
                self._set_bn(layer.src, layer.dst)
            if layer.type == 'fc':
                self._set_fc(layer.src, layer.dst)

        return self.state_dict
