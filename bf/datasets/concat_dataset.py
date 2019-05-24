import types

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset as TorchConcatDataset

import bf.datasets
from bf.utils.misc_utils import filter_kwargs


class ConcatDataset(TorchDataset):
    def __init__(self, datasets, labels, label_map={}, augment=None, preprocess=None):
        self.class_labels = ['background'] + list(labels)
        self.num_classes = len(self.class_labels)

        dataset_iter = iter(datasets)
        datasets = []

        collate = None
        display = None

        for dataset_args in dataset_iter:
            Dataset = getattr(bf.datasets, dataset_args['name'])

            if collate:
                assert collate is Dataset.collate
            else:
                collate = Dataset.collate

            if display:
                assert display is Dataset.display
            else:
                display = Dataset.display

            Dataset = filter_kwargs(Dataset)
            kwargs = dict(dataset_args.items())
            kwargs.update({
                'labels': labels,
                'label_map': label_map,
                'augment': augment,
                'preprocess': preprocess
            })
            datasets.append(Dataset(**kwargs))

        self.dataset = TorchConcatDataset(datasets)

    def collate(self, *args, **kwargs):
        return self.dataset.datasets[0].collate(*args, **kwargs)

    def display(self, *args, **kwargs):
        return self.dataset.datasets[0].display.__func__(self, *args, **kwargs)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
