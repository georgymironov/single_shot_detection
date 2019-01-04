import torch
import torch.nn as nn
from torch.nn import NLLLoss, SmoothL1Loss


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=None, reduction='mean'):
        super(SoftmaxFocalLoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError('Wrong value for reduction: {reduction}')

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input_, target):
        logpb = input_.gather(1, target.view(-1, 1)).view(-1)
        pb = logpb.exp()

        loss = -1 * (1 - pb).pow_(self.gamma).mul_(logpb)

        if self.alpha is not None:
            alpha = torch.full_like(target, self.alpha, dtype=torch.float)
            alpha[target == 0] = 1 - self.alpha
            loss.mul_(alpha)

        if self.reduction == 'mean':
            loss.mean_()
        elif self.reduction == 'sum':
            loss.sum_()

        return loss
