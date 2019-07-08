import torch
import torch.nn as nn

from bf.modules import losses
from bf.utils.misc_utils import get_ctor
from detection.target_assigner import LOC_INDEX_START, LOC_INDEX_END, CLASS_INDEX, SCORE_INDEX, IGNORE_INDEX


class MultiboxLoss(nn.Module):
    def __init__(self,
                 classification_loss,
                 localization_loss,
                 sampler,
                 classification_weight=1.0,
                 localization_weight=1.0):
        super(MultiboxLoss, self).__init__()

        ClassificationLoss = get_ctor(losses, classification_loss['name'])
        self.classification_loss = ClassificationLoss(reduction='sum', ignore_index=IGNORE_INDEX, **classification_loss)
        self.soft_target = getattr(self.classification_loss, 'SOFT_TARGET', False)
        self.multiclass = getattr(self.classification_loss, 'MULTICLASS', False)

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
            target: torch.tensor(:shape [Batch, AnchorBoxes, 6])
        Returns:
            losses: tuple(float, float)
        """
        scores, locs = pred

        target_locs = target[..., LOC_INDEX_START:LOC_INDEX_END]
        target_classes = target[..., CLASS_INDEX].long()
        target_scores = target[..., SCORE_INDEX]

        batch_size = target.size(0)
        num_priors = target.size(1)

        scores = scores.view(batch_size, num_priors, -1)
        locs = locs.view(batch_size, num_priors, 4)

        positive_mask = target_classes.gt(0)
        sampled_mask = self.sampler(scores, target_classes)

        scores = scores[sampled_mask]
        target_classes = target_classes[sampled_mask]
        target_scores = target_scores[sampled_mask]

        if self.multiclass:
            class_target = torch.zeros_like(scores)
            mask = target_classes != 0
            class_target[mask, target_classes[mask] - 1] = target_scores[mask]
        elif self.soft_target:
            class_target = torch.zeros_like(scores)
            class_target[torch.ones(class_target.size(0), dtype=torch.uint8, device=target.device), target_classes] = target_scores
        else:
            class_target = target_classes.view(-1)

        class_loss = self.classification_loss(scores, class_target)

        positive_locs = locs[positive_mask].view(-1, 4)
        positive_target_locs = target_locs[positive_mask].view(-1, 4)
        loc_loss = self.localization_loss(positive_locs, positive_target_locs)

        divider = positive_mask.sum().clamp_(min=1).float()
        loc_loss.mul_(self.localization_weight).div_(divider)
        class_loss.mul_(self.classification_weight).div_(divider)

        loss = class_loss + loc_loss

        return loss, class_loss, loc_loss
