import numpy as np
import torch
from torch.utils.data import DataLoader

import bf.datasets
from bf.preprocessing import transforms


def create_dataloaders(dataset_params,
                       batch_size,
                       input_size,
                       augmentations=[],
                       preprocessing=[],
                       shuffle=False,
                       num_workers=0,
                       pin_memory=False):

    resize = transforms.Resize(input_size)
    augment, preprocess = [
        transforms.Compose([getattr(transforms, x['name'])(**x.get('args', {})) for x in transform_set])
        for transform_set in (augmentations, preprocessing)
    ]

    seed = torch.initial_seed()
    dataset = {}
    dataloader = {}

    num_classes = None
    for phase in dataset_params.keys():
        Dataset = getattr(bf.datasets, dataset_params[phase]['name'])
        kwargs = {k: v for k, v in dataset_params[phase].items() if k in Dataset.__init__.__code__.co_varnames}

        if phase == 'train':
            dataset[phase] = Dataset(**kwargs,
                                     resize=resize,
                                     augment=augment,
                                     preprocess=preprocess)

        if phase == 'val':
            dataset[phase] = Dataset(**kwargs,
                                     resize=resize,
                                     preprocess=preprocess)

        dataloader[phase] = DataLoader(dataset[phase],
                                       batch_size=batch_size,
                                       shuffle=shuffle and phase == 'train',
                                       collate_fn=dataset[phase].collate,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory,
                                       worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
        if num_classes is None:
            num_classes = dataset[phase].num_classes
        else:
            assert num_classes == dataset[phase].num_classes

    return dataloader, num_classes, dataset
