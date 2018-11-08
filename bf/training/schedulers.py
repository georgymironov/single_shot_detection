from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import *


class LinearGrowthLR(_LRScheduler):
    def __init__(self, optimizer, cold_lr, steps, last_epoch=-1):
        if not isinstance(cold_lr, list):
            cold_lr = [cold_lr] * len(optimizer.param_groups)

        self.cold_lr = cold_lr
        self.steps = steps
        super(LinearGrowthLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [blr + self.last_epoch * (lr - blr) / (self.steps - 1) for blr, lr in zip(self.cold_lr, self.base_lrs)]

class CosineAnnealingWithWarmupLR(object):
    def __new__(cls, optimizer, T_max, eta_min=0.0, warmup_steps=0, warmup_lr=0.0, last_epoch=-1):
        schedulers = [
            (0, 'LinearGrowthLR', {'cold_lr': warmup_lr, 'steps': warmup_steps}),
            (warmup_steps, 'CosineAnnealingLR', {'T_max': T_max - warmup_steps, 'eta_min': eta_min})
        ]
        return ConcatScheduler(optimizer, schedulers, last_epoch=last_epoch)

class ConcatScheduler(object):
    def __init__(self, optimizer, schedulers, last_epoch=-1):
        self.schedulers = [(start_epoch, globals()[name](optimizer,
                                                         last_epoch=max(last_epoch - start_epoch, -1),
                                                         **args))
                           for start_epoch, name, args in schedulers]

        self.scheduler_idx = 0
        self.last_epoch = last_epoch
        self._next_scheduler()

    @property
    def current_scheduler(self):
        return self.schedulers[self.scheduler_idx][1]

    @property
    def current_scheduler_start_epoch(self):
        return self.schedulers[self.scheduler_idx][0]

    def _next_scheduler(self):
        while (self.scheduler_idx < len(self.schedulers) - 1 and
               self.last_epoch + 1 >= self.schedulers[self.scheduler_idx + 1][0]):
            self.scheduler_idx += 1

    def get_lr(self):
        return self.current_scheduler.get_lr()

    def step(self, epoch=None):
        self._next_scheduler()
        if epoch is not None:
            epoch -= self.current_scheduler_start_epoch
        self.current_scheduler.step(epoch)
        self.last_epoch = self.current_scheduler.last_epoch + self.current_scheduler_start_epoch
