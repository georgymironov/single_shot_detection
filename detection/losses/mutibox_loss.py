import torch
import torch.nn as nn

from bf.modules import losses
from bf.utils.misc_utils import get_ctor


class MultiboxLoss(nn.Module):
    def __init__(self,
                 classification_loss,
                 localization_loss,
                 sampler,
                 classification_weight=1.0,
                 localization_weight=1.0):
        super(MultiboxLoss, self).__init__()

        ClassificationLoss = get_ctor(losses, classification_loss['name'])
        self.classification_loss = ClassificationLoss(reduction='sum', ignore_index=-1, **classification_loss)

        LocalizationLoss = get_ctor(losses, localization_loss['name'])
        self.localization_loss = LocalizationLoss(reduction='sum', **localization_loss)

        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.sampler = sampler

    def forward(self, pred, target):
        """
        Args:
            pred: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes * Classes])
                torch.tensor(:shape [Batch, AnchorBoxes * 4])
            target: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes])
                torch.tensor(:shape [Batch, AnchorBoxes, 4])
        Returns:
            losses: tuple(float, float)
        """
        classes, locs = pred
        target_classes, target_locs = target

        batch_size = classes.size(0)
        num_priors = target_classes.size(1)

        classes = classes.view(batch_size, num_priors, -1)
        locs = locs.view(batch_size, num_priors, 4)

        mask = self.sampler(classes, target_classes)
        class_loss = self.classification_loss(classes[mask], target_classes[mask].view(-1))

        positive_mask = target_classes.gt(0)
        positive_locs = locs[positive_mask].view(-1, 4)
        positive_target_locs = target_locs[positive_mask].view(-1, 4)
        loc_loss = self.localization_loss(positive_locs, positive_target_locs)

        divider = positive_mask.sum().clamp_(min=1).float()
        loc_loss.mul_(self.localization_weight).div_(divider)
        class_loss.mul_(self.classification_weight).div_(divider)

        loss = class_loss + loc_loss

        return loss, class_loss, loc_loss
