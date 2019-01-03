import torch


def naive_sampler(loss, target_classes):
    return torch.ones_like(loss, dtype=torch.uint8)

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
