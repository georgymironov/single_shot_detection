import logging

import torch


class from_keras(object):
    state_dict = None

    def __init__(self, model):
        self.model = model
        self.source_layers = [x.name for x in model.layers]

    def to(self, state_dict):
        self.state_dict = state_dict
        return self

    def _get_weights(self, name):
        if name not in self.source_layers:
            logging.warning(f'{name} layer is missing in source model')
            return []

        layer = self.model.get_layer(name)
        return zip([x.name for x in layer.weights],
                   layer.get_weights())

    def _set_weight(self, weight, src, dst, permute=None):
        if dst in self.state_dict:
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

    def mobilenet_v2(self):
        self._set_conv(src='Conv1', dst='features.0.conv')
        self._set_bn(src='bn_Conv1', dst='features.0.bn')
        self._set_conv(src='expanded_conv_depthwise', dst='features.1.depthwise_conv')
        self._set_bn(src='expanded_conv_depthwise_BN', dst='features.1.depthwise_bn')
        self._set_conv(src='expanded_conv_project', dst='features.1.project_conv')
        self._set_bn(src='expanded_conv_project_BN', dst='features.1.project_bn')

        for i in range(2, 18):
            self._set_conv(src=f'block_{i-1}_expand', dst=f'features.{i}.expand_conv')
            self._set_bn(src=f'block_{i-1}_expand_BN', dst=f'features.{i}.expand_bn')
            self._set_conv(src=f'block_{i-1}_depthwise', dst=f'features.{i}.depthwise_conv')
            self._set_bn(src=f'block_{i-1}_depthwise_BN', dst=f'features.{i}.depthwise_bn')
            self._set_conv(src=f'block_{i-1}_project', dst=f'features.{i}.project_conv')
            self._set_bn(src=f'block_{i-1}_project_BN', dst=f'features.{i}.project_bn')

        self._set_conv(src='Conv_1', dst='features.18.conv')
        self._set_bn(src='Conv_1_bn', dst='features.18.bn')

        if 'logits.weight' in self.state_dict:
            self._set_fc(src='Logits', dst='logits')

        return self.state_dict
