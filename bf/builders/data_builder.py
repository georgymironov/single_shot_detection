import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import bf.datasets
import bf.preprocessing
from bf.preprocessing import transforms


def create_preprocessing(input_size, augmentations, preprocessing, transform_type):
    bf.preprocessing.set_transform_type(transform_type)
    resize = bf.preprocessing.transforms.Resize(input_size)
    augment, preprocess = [
        Compose([getattr(bf.preprocessing.transforms, x['name'])(**x.get('args', {})) for x in transform_set])
        for transform_set in (augmentations, preprocessing)
    ]
    return resize, augment, preprocess

def create_dataloaders(data_params,
                       batch_size,
                       resize,
                       augment,
                       preprocess,
                       shuffle=False,
                       num_workers=0,
                       pin_memory=False):
    def _build_dataloader(dataset_params,
                          phase,
                          labels=None,
                          label_map={}):
        Dataset = getattr(bf.datasets, dataset_params['name'])

        kwargs = dict(dataset_params.items())
        kwargs.update({'labels': labels, 'label_map': label_map})
        kwargs = {k: v for k, v in kwargs.items() if k in Dataset.__init__.__code__.co_varnames}

        dataset = Dataset(**kwargs,
                          resize=resize,
                          augment=augment if phase == 'train' else None,
                          preprocess=preprocess)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size * 2 if phase == 'eval' else batch_size,
                                shuffle=shuffle and phase == 'train',
                                collate_fn=dataset.collate,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

        return dataset, dataloader

    seed = torch.initial_seed()
    dataset = {}
    dataloader = {}

    for phase in ['train', 'eval']:
        if phase in data_params:
            dataset[phase], dataloader[phase] = _build_dataloader(data_params[phase],
                                                                  phase,
                                                                  labels=data_params.get('labels'),
                                                                  label_map=data_params.get('label_map', {}))

    return dataloader, dataset
