import logging
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bf.datasets
from bf.preprocessing import transforms


def create_preprocessing(augmentations, preprocessing, input_size=None, transform_type='no_target'):
    augment = transforms.Compose(augmentations, transform_type=transform_type)
    preprocess = transforms.Compose(preprocessing, transform_type=transform_type)

    if input_size:
        resize = transforms.Resize(input_size, transform_type=transform_type)
        preprocess.transforms.insert(0, resize)

    return augment, preprocess

def _worker_init_fn(worker_id, seed):
    import numpy as np
    np.random.seed(seed + worker_id)

def create_dataloaders(data_params,
                       batch_size,
                       augment,
                       preprocess,
                       shuffle=False,
                       num_workers=0,
                       pin_memory=False,
                       distributed=False):

    if distributed and shuffle:
        logging.warn('WRN: dataset shuffling is not supported for distributed training')

    worker_init_fn = partial(_worker_init_fn, seed=torch.initial_seed())

    def _build_dataloader(dataset_params,
                          phase,
                          labels=None,
                          label_map={}):
        Dataset = getattr(bf.datasets, dataset_params['name'])

        kwargs = dict(dataset_params.items())
        kwargs.update({'labels': labels, 'label_map': label_map})
        kwargs = {k: v for k, v in kwargs.items() if k in Dataset.__init__.__code__.co_varnames}

        dataset = Dataset(**kwargs,
                          augment=augment if phase == 'train' else None,
                          preprocess=preprocess)

        if distributed:
            sampler = DistributedSampler(dataset)
        elif shuffle and phase == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size * 2 if phase == 'eval' else batch_size,
                                sampler=sampler,
                                collate_fn=dataset.collate,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=True,
                                worker_init_fn=worker_init_fn)

        return dataset, dataloader

    datasets = {}
    dataloaders = {}

    for phase in ['train', 'eval']:
        if phase in data_params:
            datasets[phase], dataloaders[phase] = _build_dataloader(data_params[phase],
                                                                    phase,
                                                                    labels=data_params.get('labels'),
                                                                    label_map=data_params.get('label_map', {}))

    return dataloaders, datasets
