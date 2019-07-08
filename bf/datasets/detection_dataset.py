import logging

from jpeg4py import JPEG
import torch
from torch.utils.data import Dataset

from bf.utils import image_utils


LOC_INDEX_START = 0
LOC_INDEX_END = 4
CLASS_INDEX = 4
SCORE_INDEX = 5
DIFFICULT_INDEX = 6

NEGATIVE_CLASS = 0


class DetectionDataset(Dataset):
    def __getitem__(self, index):
        annotation = self.annotations[index]
        img = JPEG(annotation['image_path']).decode()
        target = annotation['boxes'].copy()

        if self.augment:
            img, target = self.augment((img, target))
        if self.preprocess:
            img, target = self.preprocess((img, target))

        valid_idx = (target[..., [0, 1]] != target[..., [2, 3]]).all(1)
        target = target[valid_idx]
        if (target[..., [0, 1]] > target[..., [2, 3]]).any():
            logging.warn(f'WW Invalid values for target: {annotation["image_path"]}')
        if (target[..., :4] < 0).any():
            logging.warn(f'WW Negative values for target: {annotation["image_path"]}')

        return img, target

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def collate(batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        targets = list(targets)
        return imgs, targets

    def display(self, index):
        image_utils.display(*self[index])
