import numpy as np
import torch

from bf.core.target_types import TargetTypes
from bf.datasets import detection_dataset as det_ds


class BatchContainer(object):
    def __init__(self, batch, target_type):
        imgs, targets = zip(*batch)

        self.imgs = torch.stack(imgs, dim=0)

        if target_type == TargetTypes.Boxes:
            self.targets = targets
        else:
            raise ValueError(f'Unknown type {target_type}')

        self.target_type = target_type

    def to_(self, device):
        self.imgs = self.imgs.to(device)
        return self

    def mixup_(self, alpha, p):
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(self.imgs.size(0))
        roll = torch.rand(self.imgs.size(0)) < p

        self.imgs[roll] = lam * self.imgs[roll] + (1.0 - lam) * self.imgs[index][roll]

        if self.target_type == TargetTypes.Boxes:
            mixed_t = []
            for i, target in enumerate(self.targets):
                if not roll[i]:
                    mixed_t.append(target)
                    continue
                original = target.clone()
                original[..., det_ds.SCORE_INDEX] *= lam
                mixed = self.targets[index[i]].clone()
                mixed[..., det_ds.SCORE_INDEX] *= (1.0 - lam)
                mixed_t.append(torch.cat([original, mixed], dim=0))
            self.targets = mixed_t

        return self

    def pin_memory(self):
        self.imgs = self.imgs.pin_memory()

        if self.target_type == TargetTypes.Boxes:
            self.targets = [t.pin_memory() for t in self.targets]

        return self

    def get(self):
        return self.imgs, self.targets
