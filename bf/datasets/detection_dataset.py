from jpeg4py import JPEG

import torch
from torch.utils.data import Dataset

from bf.utils import image_utils


class DetectionDataset(Dataset):
    def __getitem__(self, index):
        annotation = self.annotations[index]
        img = JPEG(annotation['image_path']).decode()
        target = annotation['boxes'].copy()

        if self.augment:
            img, target = self.augment((img, target))
        if self.resize:
            img, target = self.resize((img, target))
        if self.preprocess:
            img, target = self.preprocess((img, target))

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
