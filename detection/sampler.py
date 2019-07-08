import math

import torch
import torch.nn.functional as F

from detection.target_assigner import NEGATIVE_CLASS, IGNORE_CLASS


def naive_sampler(predictions, target_classes):
    return target_classes.ne(NEGATIVE_CLASS) & target_classes.ne(IGNORE_CLASS)

def hard_negative_mining(predictions, target_classes, negative_per_positive_ratio, min_negative_per_image):
    loss = -F.log_softmax(predictions, dim=-1)[:, :, NEGATIVE_CLASS]

    negative_mask = target_classes.eq(NEGATIVE_CLASS)
    positive_mask = target_classes.ne(NEGATIVE_CLASS) & target_classes.ne(IGNORE_CLASS)
    num_negatives = negative_mask.sum(dim=1, keepdim=True)
    num_positives = positive_mask.sum(dim=1, keepdim=True)

    num_negatives = torch.min(torch.clamp_(num_positives * negative_per_positive_ratio, min=min_negative_per_image), num_negatives)
    loss[~negative_mask] = -math.inf
    rank = loss.argsort(dim=1, descending=True).argsort(dim=1)
    hard_negative_mask = rank < num_negatives

    return positive_mask | hard_negative_mask
