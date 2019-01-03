import torch
import torch.nn as nn

from bf.modules import losses


def hard_negative_mining(loss, target_classes, negative_per_positive_ratio, min_negative_per_image):
    negative_mask = target_classes.eq(0)
    positive_mask = target_classes.gt(0)
    num_negatives = negative_mask.sum(dim=1, keepdim=True)
    num_positives = positive_mask.sum(dim=1, keepdim=True)

    num_negatives = torch.min(torch.clamp_(num_positives * negative_per_positive_ratio, min=min_negative_per_image), num_negatives)
    negative_loss = loss * negative_mask.float()
    rank = negative_loss.argsort(dim=1, descending=True).argsort(dim=1)
    hard_negative_mask = rank < num_negatives

    return positive_mask | hard_negative_mask

class MultiboxLoss(nn.Module):
    def __init__(self,
                 classification_loss,
                 localization_loss,
                 negative_per_positive_ratio,
                 classification_weight=1.0,
                 localization_weight=1.0,
                 min_negative_per_image=0):
        super(MultiboxLoss, self).__init__()

        ClassificationLoss = getattr(losses, classification_loss['name'])
        kwargs = {k: v for k, v in classification_loss.items() if k in ClassificationLoss.__init__.__code__.co_varnames}
        self.classification_loss = ClassificationLoss(reduction='none', ignore_index=-1, **kwargs)

        LocalizationLoss = getattr(losses, localization_loss['name'])
        kwargs = {k: v for k, v in localization_loss.items() if k in LocalizationLoss.__init__.__code__.co_varnames}
        self.localization_loss = LocalizationLoss(reduction='sum', **kwargs)

        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.negative_per_positive_ratio = negative_per_positive_ratio
        self.min_negative_per_image = min_negative_per_image

    def forward(self, pred, target):
        """
        Args:
            pred: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes, Classes])
                torch.tensor(:shape [Batch, AnchorBoxes, 4])
                torch.tensor(:shape [AnchorBoxes, 4])
            target: tuple of
                torch.tensor(:shape [Batch, AnchorBoxes, 4])
                torch.tensor(:shape [Batch, AnchorBoxes])
        Returns:
            losses: tuple(float, float)
        """
        classes, locs = pred
        batch_size = classes.size(0)
        num_classes = classes.size(2)
        target_classes, target_locs = target

        classes = classes.view(-1, num_classes)
        class_loss = self.classification_loss(classes, target_classes.view(-1))

        class_loss = class_loss.view(batch_size, -1)
        mask = hard_negative_mining(class_loss, target_classes, self.negative_per_positive_ratio, self.min_negative_per_image)
        class_loss = torch.sum(class_loss[mask])

        positive_mask = target_classes.gt(0)
        positive_locs = locs[positive_mask].view(-1, 4)
        positive_target_locs = target_locs[positive_mask].view(-1, 4)
        loc_loss = self.localization_loss(positive_locs, positive_target_locs)

        divider = positive_mask.sum().clamp(min=1).float()
        loc_loss /= divider
        class_loss /= divider

        loss = self.classification_weight * class_loss + self.localization_weight * loc_loss

        return loss, class_loss, loc_loss
