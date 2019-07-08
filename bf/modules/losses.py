import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import *


class _Loss(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.0):
        super(_Loss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Wrong value for reduction: {reduction}')

        assert 0.0 <= epsilon < 1

        self.reduction = reduction
        self.epsilon = epsilon

    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def _soften(self, target):
        pos = target.gt(0).float()
        target += (1.0 - pos) * self.epsilon * target.sum(-1, keepdim=True) / (target.size(1) - pos.sum(-1, keepdim=True))
        target -= pos * self.epsilon * target
        return target

class SigmoidFocalLoss(_Loss):
    MULTICLASS = True

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(SigmoidFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, prediction, target):
        loss = torch.zeros(target.size(0), dtype=torch.float32, device=target.device)

        alpha_weight = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        pb = torch.sigmoid(prediction)
        pb = pb * target + (1 - pb) * (1.0 - target)

        cross_entropy = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')

        loss = (alpha_weight * (1 - pb).pow(self.gamma) * cross_entropy).sum(dim=-1)

        return self._reduce(loss)

class SoftmaxFocalLoss(_Loss):
    def __init__(self, gamma=0.0, alpha=None, ignore_index=-100, **kwargs):
        super(SoftmaxFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
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

        return self._reduce(loss)

class CrossEntropyWithSoftTargetsLoss(_Loss):
    SOFT_TARGET = True

    def forward(self, logits, target):
        loss = torch.zeros(target.size(0), dtype=torch.float32, device=target.device)

        logpb = F.log_softmax(logits, dim=-1)

        if self.epsilon:
            target = self._soften(target)

        loss = -1 * logpb.mul(target).sum(dim=-1)

        return self._reduce(loss)

class BinaryCrossEntropyWithSoftTargetsLoss(_Loss):
    SOFT_TARGET = True
    MULTICLASS = True

    def forward(self, logits, target):
        if self.epsilon:
            target = self._soften(target)

        return F.binary_cross_entropy_with_logits(logits, target, reduction=self.reduction)
