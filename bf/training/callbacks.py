import os
import shutil

import pandas as pd
from tensorboardX import SummaryWriter
import torch

from bf.training import schedulers


def checkpoint(event_emitter, checkpoint_dir, config_path=None, save_every=1):
    os.path.exists(checkpoint_dir) or os.makedirs(checkpoint_dir)
    new_config_path = os.path.join(checkpoint_dir, 'config.py')
    if os.path.exists(config_path):
        if not os.path.exists(new_config_path) or not os.path.samefile(config_path, new_config_path):
            shutil.copy(config_path, new_config_path)
    print(f'===> Checkpoints will be saved to {checkpoint_dir}')

    @event_emitter.on('epoch_end')
    def save_checkpoint(global_state=None, **kwargs):
        if (global_state['epoch'] + 1) % save_every == 0:
            torch.save(global_state, os.path.join(checkpoint_dir, f'ckpt-{global_state["global_step"]}.pt'))

def logger(event_emitter, log_dir):
    log_path = os.path.join(log_dir, 'log.txt')
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, index_col=['global_step'])
    else:
        log = pd.DataFrame()
        log.index.name = 'global_step'

    @event_emitter.on('epoch_end')
    def log_fn(global_state=None, epoch_state=None, **kwargs):
        nonlocal log
        log = log.append(pd.Series(epoch_state, name=global_state['global_step']))
        with open(log_path, 'w') as f:
            f.write(log.to_csv())

def tensorboard(event_emitter, log_dir):
    writer = SummaryWriter(log_dir)

    @event_emitter.on('step_end')
    def log_train_state(phase=None, global_state=None, state=None, **kwargs):
        if phase == 'train':
            for k, v in state.items():
                writer.add_scalar(f'train/{k}', v, global_state['global_step'])

    @event_emitter.on('phase_end')
    def log_phase_state(phase=None, global_state=None, phase_state=None, **kwargs):
        for k, v in phase_state.items():
            writer.add_scalar(f'{phase}/{k}', v, global_state['global_step'])

    return writer

def scheduler(event_emitter, scheduler_, run_scheduler_each_step, scheduler_metric):
    def scheduler_step(phase=None, global_state=None, phase_state=None, *args, **kwargs):
        if phase == 'train' and not isinstance(scheduler_, schedulers.ReduceLROnPlateau):
            scheduler_.step()
            event_emitter.emit('scheduler_step')
        if phase == 'eval' and isinstance(scheduler_, schedulers.ReduceLROnPlateau):
            scheduler_.step(phase_state[scheduler_metric])
            event_emitter.emit('scheduler_step')

    if run_scheduler_each_step:
        event_name = 'step_start'
    elif isinstance(scheduler_, schedulers.ReduceLROnPlateau):
        event_name = 'phase_end'
    else:
        event_name = 'phase_start'

    event_emitter.add_event_handler(event_name, scheduler_step)
