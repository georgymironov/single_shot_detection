import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import *


def _soften(target, epsilon):
    pos = target.gt(0).float()
    target += (1.0 - pos) * epsilon * target.sum(-1, keepdim=True) / (target.size(1) - pos.sum(-1, keepdim=True))
    target -= pos * epsilon * target
    return target

class SigmoidFocalLoss(nn.Module):
    MULTICLASS = True

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', ignore_index=-100):
        super(SigmoidFocalLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Wrong value for reduction: {reduction}')

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        loss = torch.zeros(target.size(0), dtype=torch.float32, device=target.device)
        mask = target.ne(self.ignore_index).all(dim=-1)

        prediction = prediction[mask]
        target = target[mask]

        alpha_weight = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        pb = torch.sigmoid(prediction)
        pb = pb * target + (1 - pb) * (1.0 - target)

        cross_entropy = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')

        loss[mask] = (alpha_weight * (1 - pb).pow(self.gamma) * cross_entropy).sum(dim=-1)

        if self.reduction == 'mean':
            loss = loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()

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
            loss = loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class CrossEntropyWithSoftTargetsLoss(nn.Module):
    SOFT_TARGET = True

    def __init__(self, reduction='mean', ignore_index=-100, epsilon=0.0):
        super(CrossEntropyWithSoftTargetsLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Wrong value for reduction: {reduction}')

        assert 0.0 <= epsilon < 1

        self.reduction = reduction
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits, target):
        loss = torch.zeros(target.size(0), dtype=torch.float32, device=target.device)
        mask = target.ne(self.ignore_index).all(dim=-1)

        logpb = F.log_softmax(logits[mask], dim=-1)
        target = target[mask]

        if self.epsilon:
            target = _soften(target, self.epsilon)

        loss[mask] = -1 * logpb.mul(target).sum(dim=-1)

        if self.reduction == 'mean':
            loss = loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class BinaryCrossEntropyWithSoftTargetsLoss(nn.Module):
    SOFT_TARGET = True
    MULTICLASS = True

    def __init__(self, reduction='mean', ignore_index=-100, epsilon=0.0):
        super(BinaryCrossEntropyWithSoftTargetsLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Wrong value for reduction: {reduction}')

        assert 0.0 <= epsilon < 1

        self.reduction = reduction
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits, target):
        loss = torch.zeros(target.size(0), dtype=torch.float32, device=target.device)
        mask = target.ne(self.ignore_index).all(dim=-1)

        logits = logits[mask]
        target = target[mask]

        if self.epsilon:
            target = _soften(target, self.epsilon)

        loss[mask] = F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum(dim=-1)

        if self.reduction == 'mean':
            loss = loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
