import logging

import torch

from bf.training import optimizers, schedulers


def create_optimizer(model,
                     optimizer_params,
                     state={}):

    if hasattr(optimizer_params, 'lr_groups'):
        parameters = [{
            'params': getattr(model, name).parameters(),
            'lr': lr
        } for name, lr in optimizer_params.lr_groups.items()]
    else:
        parameters = model.parameters()

    Optimizer = getattr(optimizers, optimizer_params['name'])
    kwargs = {k: v for k, v in optimizer_params.items() if k in Optimizer.__init__.__code__.co_varnames}
    optimizer = Optimizer(parameters, **kwargs)
    if 'optimizer_dict' in state:
        logging.info('===> Loading optimizer weights from checkpoint')
        optimizer.load_state_dict(state['optimizer_dict'])
        del state['optimizer_dict']
        torch.cuda.empty_cache()

    logging.info(optimizer)

    return optimizer

def create_scheduler(scheduler_params, optimizer, state={}):
    run_scheduler_each_step = scheduler_params.get('run_each_step', False)
    scheduler_metric = scheduler_params.get('scheduler_metric', 'eval_loss')

    Scheduler = getattr(schedulers, scheduler_params['name'])
    kwargs = {k: v for k, v in scheduler_params.items() if k in Scheduler.__init__.__code__.co_varnames}

    if Scheduler is schedulers.ReduceLROnPlateau:
        scheduler = Scheduler(optimizer, **kwargs)
    else:
        last_epoch = state.get('global_step', -1) if run_scheduler_each_step else state.get('epoch', -1)
        logging.info(f'===> Setting scheduler "last_epoch" to {last_epoch}')
        last_epoch = last_epoch + 1 if last_epoch >= 0 else last_epoch  # wtf
        scheduler = Scheduler(optimizer, last_epoch=last_epoch, **kwargs)

    return scheduler, run_scheduler_each_step, scheduler_metric
