import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, SmoothL1Loss, NLLLoss


def _one_hot(prediction, target):
    mask = target != 0
    target_ = torch.zeros_like(prediction)
    target_[mask, target[mask] - 1] = 1
    return target_

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', ignore_index=-100):
        super(SigmoidFocalLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Wrong value for reduction: {reduction}')

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        loss = torch.zeros(target.size(), dtype=torch.float32, device=target.device)

        mask = target != self.ignore_index
        prediction = prediction[mask]
        target = target[mask]

        target = _one_hot(prediction, target)

        alpha_weight = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        pb = torch.sigmoid(prediction)
        pb = (1 - pb) * target + pb * (1.0 - target)

        cross_entropy = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')

        loss[mask] = (alpha_weight * pb.pow(self.gamma) * cross_entropy).sum(dim=-1)

        if self.reduction == 'mean':
            loss.mean_()
        elif self.reduction == 'sum':
            loss.sum_()

        return loss

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=None, reduction='mean', ignore_index=-100):
        super(SoftmaxFocalLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Wrong value for reduction: {reduction}')

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input_, target):
        loss = torch.zeros(target.size(), dtype=torch.float32, device=target.device)
        mask = target != self.ignore_index

        logpb = F.log_softmax(input_, dim=-1)
        logpb = logpb[mask, target[mask]]
        pb = logpb.exp()

        loss[mask] = -1 * (1 - pb).pow_(self.gamma).mul_(logpb)

        if self.alpha is not None:
            alpha = torch.full_like(target, self.alpha, dtype=torch.float)
            alpha[target == 0] = 1 - self.alpha
            loss.mul_(alpha)

        if self.reduction == 'mean':
            loss.mean_()
        elif self.reduction == 'sum':
            loss.sum_()

        return loss
