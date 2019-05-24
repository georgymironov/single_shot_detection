import os

import torch
import torch.nn as nn

from bf import base as detection_base


class _resnet_wrapper(nn.Module):
    def __init__(self, model):
        super(_resnet_wrapper, self).__init__()

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x):
        return self.features(x)

def create_base(model_params):
    Base = getattr(detection_base, model_params['base']['name'])
    kwargs = {}

    weight = model_params['base'].get('weight', None)
    if weight == 'torchvision':
        kwargs['pretrained'] = True

    batch_norm = model_params['base'].get('batch_norm', {})
    if batch_norm:
        kwargs['batch_norm_params'] = batch_norm

    base = Base(**kwargs)

    if weight == 'torchvision' and model_params['base']['name'].startswith('resnet'):
        base = _resnet_wrapper(base)

    if weight == 'keras':
        base.init_from_keras()
    elif weight is not None and os.path.exists(weight):
        base.load_state_dict(torch.load(weight), map_location='cpu')

    return base
