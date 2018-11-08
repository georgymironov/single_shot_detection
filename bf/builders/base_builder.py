import os

import torch

from bf import base as detection_base


def create_base(model_params):
    Base = getattr(detection_base, model_params['base']['name'])
    kwargs = {'init_weights': False}

    weight = model_params['base'].get('weight', None)
    if weight is None:
        kwargs['init_weights'] = True
    elif weight == 'torchvision':
        kwargs['pretrained'] = True

    batch_norm = model_params['base'].get('batch_norm', {})
    if batch_norm:
        kwargs['batch_norm_params'] = batch_norm

    base = Base(**kwargs)

    if weight == 'keras':
        base.init_from_keras()
    elif os.path.exists(weight):
        base.load_state_dict(torch.load(weight))

    return base
