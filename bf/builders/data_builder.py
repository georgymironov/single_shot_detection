import logging
from copy import copy
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import bf.datasets
from bf.core.target_types import TargetTypes
from bf.preprocessing import transforms
from bf.utils.misc_utils import get_ctor


def create_preprocessing(augmentations, preprocessing, input_size=None, transform_type=TargetTypes.NoTarget):
    augment = transforms.Compose(augmentations, transform_type=transform_type)
    preprocess = transforms.Compose(preprocessing, transform_type=transform_type)

    if input_size:
        resize = transforms.Resize(input_size, transform_type=transform_type)
        preprocess.transforms.insert(0, resize)

    return augment, preprocess

def create_datasets(datasets, augment, preprocess):
    _datasets = {}
    labels = datasets.get('labels')
    label_map = datasets.get('label_map', {})

    for phase in ['train', 'eval']:
        if phase not in datasets:
            continue
        Dataset = get_ctor(bf.datasets, datasets[phase]['name'])
        kwargs = copy(datasets[phase])
        kwargs.update({'labels': labels, 'label_map': label_map})
        _datasets[phase] = Dataset(**kwargs,
                                   augment=augment if phase == 'train' else None,
                                   preprocess=preprocess)

    return _datasets

def create_samples(datasets,
                   shuffle=False,
                   distributed=False):
    if distributed and shuffle:
        logging.warn('WRN: dataset shuffling is not supported for distributed training')

    samplers = {}

    for phase in datasets.keys():
        if distributed and phase == 'train':
            samplers[phase] = DistributedSampler(datasets[phase])
        elif shuffle and phase == 'train':
            samplers[phase] = RandomSampler(datasets[phase])
        else:
            samplers[phase] = SequentialSampler(datasets[phase])

    return samplers

def _worker_init_fn(worker_id, seed):
    import numpy as np
    np.random.seed(seed + worker_id)

def create_dataloaders(datasets,
                       samplers,
                       batch_size,
                       num_workers=0,
                       pin_memory=False):
    dataloaders = {}
    worker_init_fn = partial(_worker_init_fn, seed=torch.initial_seed())

    for phase in datasets.keys():
        _batch_size = batch_size * 2 if phase == 'eval' else batch_size
        _drop_last = phase == 'train'

        dataloaders[phase] = DataLoader(datasets[phase],
                                        batch_size=_batch_size,
                                        sampler=samplers[phase],
                                        collate_fn=datasets[phase].collate,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory,
                                        drop_last=_drop_last,
                                        worker_init_fn=worker_init_fn)

    return dataloaders
