from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset as TorchConcatDataset

import bf.datasets
from bf.utils.misc_utils import filter_ctor_args


class ConcatDataset(TorchDataset):
    def __init__(self, datasets, labels, label_map=None, resize=None, augment=None, preprocess=None):
        self.class_labels = ['background'] + list(labels)
        self.num_classes = len(self.class_labels)

        dataset_iter = iter(datasets)
        datasets = []
        for dataset_args in dataset_iter:
            Dataset = getattr(bf.datasets, dataset_args['name'])

            if hasattr(self, 'collate'):
                assert self.collate is Dataset.collate
            else:
                self.collate = Dataset.collate

            Dataset = filter_ctor_args(Dataset)
            kwargs = dict(dataset_args.items())
            kwargs.update({
                'labels': labels,
                'label_map': label_map,
                'resize': resize,
                'augment': augment,
                'preprocess': preprocess
            })
            datasets.append(Dataset(**kwargs))

        self.dataset = TorchConcatDataset(datasets)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
