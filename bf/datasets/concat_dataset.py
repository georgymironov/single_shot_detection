from torch.utils.data import ConcatDataset as TorchConcatDataset

import bf.datasets
from bf.datasets.detection_dataset import DetectionDataset
from bf.utils import dataset_utils


class ConcatDataset(DetectionDataset):
    def __init__(self, datasets, labels, resize=None, augment=None, preprocess=None):
        self.class_labels = ['background'] + list(labels)
        self.num_classes = len(self.class_labels)

        self.dataset = []
        for dataset_args in datasets:
            Dataset = getattr(bf.datasets, dataset_args['name'])
            dataset_args.update({'labels': labels})
            kwargs = {k: v for k, v in dataset_args.items() if k in Dataset.__init__.__code__.co_varnames}
            self.dataset.append(Dataset(**kwargs, resize=resize, augment=augment, preprocess=preprocess))
        self.dataset = TorchConcatDataset(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate(batch):
        return dataset_utils.collate_detections(batch)
