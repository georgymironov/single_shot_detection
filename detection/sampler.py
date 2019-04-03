import math

import torch
import torch.nn.functional as F


def naive_sampler(predictions, target_classes):
    return target_classes >= 0

def hard_negative_mining(predictions, target_classes, negative_per_positive_ratio, min_negative_per_image):
    loss = -F.log_softmax(predictions, dim=-1)[:, :, 0]

    negative_mask = target_classes.eq(0)
    positive_mask = target_classes.gt(0)
    num_negatives = negative_mask.sum(dim=1, keepdim=True)
    num_positives = positive_mask.sum(dim=1, keepdim=True)

    num_negatives = torch.min(torch.clamp_(num_positives * negative_per_positive_ratio, min=min_negative_per_image), num_negatives)
    loss[~negative_mask] = -math.inf
    rank = loss.argsort(dim=1, descending=True).argsort(dim=1)
    hard_negative_mask = rank < num_negatives

    return positive_mask | hard_negative_mask
