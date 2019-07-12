import os

import torch
import torch.nn as nn
import torchvision

import bf.base


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

class _shufflenet_v2_wrapper(nn.Module):
    def __init__(self, model):
        super(_shufflenet_v2_wrapper, self).__init__()

        self.features = nn.Sequential(
            model.conv1,
            model.maxpool,
            model.stage2,
            model.stage3,
            model.stage4,
            model.conv5,
        )

    def forward(self, x):
        return self.features(x)

class _senet_wrapper(nn.Module):
    def __init__(self, model):
        super(_senet_wrapper, self).__init__()

        self.features = nn.Sequential(
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x):
        return self.features(x)

def create_base(name, weight=None, **model_args):
    if name.startswith('torchhub://'):
        name = name.replace('torchhub://', '')
        repo, model = name.split(':')
        base = torch.hub.load(repo, model, **model_args)
    else:
        Base = getattr(bf.base, name)
        base = Base(**model_args)

    if isinstance(base, torchvision.models.ResNet):
        base = _resnet_wrapper(base)
    if isinstance(base, torchvision.models.ShuffleNetV2):
        base = _shufflenet_v2_wrapper(base)

    try:
        import pretrainedmodels

        if isinstance(base, pretrainedmodels.models.senet.SENet):
            base = _senet_wrapper(base)
    except Exception:
        pass

    if weight == 'keras':
        base.init_from_keras()
    elif weight is not None and os.path.exists(weight):
        base.load_state_dict(torch.load(weight, map_location='cpu'))

    return base
