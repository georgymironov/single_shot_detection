from collections import OrderedDict
import csv
import os

import torch

from bf.training import env, schedulers


@env.master_only
def progress(event_emitter):
    train_len = None

    @event_emitter.on('phase_start')
    def get_phase_len(phase, global_state, phase_len, **kwargs):
        if phase == 'train':
            nonlocal train_len
            train_len = phase_len

    @event_emitter.on('step_end')
    def print_progress(phase, global_state, state, **kwargs):
        if phase == 'train':
            messages = [f'[train] step: {global_state["global_step"]}/{train_len-1}']
            for k, v in state.items():
                messages.append(f'{k}: {v:6f}')
            for i, x in enumerate(global_state['optimizer'].param_groups):
                messages.append(f'lr {i}: {x["lr"]:.12f}')
            print(f'{", ".join(messages)}    ', end='\r')

@env.master_only
def checkpoint(event_emitter, checkpoint_dir, save_every=1):
    @event_emitter.on('epoch_end')
    def save_checkpoint(global_state=None, **kwargs):
        if os.path.exists(checkpoint_dir) and (global_state['epoch'] + 1) % save_every == 0:
            torch.save(global_state, os.path.join(checkpoint_dir, f'ckpt-{global_state["global_step"]}.pt'))

@env.master_only
def csv_logger(event_emitter, csv_log_path):
    log = []
    keys = OrderedDict({'global_step': None})

    if os.path.exists(csv_log_path):
        with open(csv_log_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames:
                keys.update(OrderedDict.fromkeys(fieldnames))
                log = list(reader)

    @event_emitter.on('epoch_end')
    def log_fn(global_state=None, epoch_state=None, **kwargs):
        nonlocal log, keys

        row = OrderedDict({'global_step': global_state['global_step']})
        row.update(epoch_state)
        log.append(row)
        keys.update(OrderedDict.fromkeys(epoch_state.keys()))

        with open(csv_log_path, 'w') as f:
            writer = csv.DictWriter(f, keys.keys())
            writer.writeheader()
            writer.writerows(log)

@env.master_only
def tensorboard(event_emitter, log_dir):
    from torch.utils.tensorboard import SummaryWriter

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

def scheduler(event_emitter, scheduler_, run_scheduler_each_step, scheduler_metric, writer=None):
    if isinstance(scheduler_, schedulers.ReduceLROnPlateau):
        def scheduler_step(phase=None, global_state=None, phase_state=None, *args, **kwargs):
            if phase == 'eval':
                scheduler_.step(phase_state[scheduler_metric])
                event_emitter.emit('scheduler_step', global_state=global_state)

        event_name = 'phase_end'
    else:
        def scheduler_step(phase=None, global_state=None, phase_state=None, *args, **kwargs):
            if phase == 'train':
                scheduler_.step()
                event_emitter.emit('scheduler_step', global_state=global_state)

        if run_scheduler_each_step:
            event_name = 'step_end'
        else:
            event_name = 'phase_end'

    event_emitter.add_event_handler(event_name, scheduler_step)

    if writer:
        @event_emitter.on('scheduler_step')
        @env.master_only
        def log_lr(global_state, **kwargs):
            for i, x in enumerate(global_state['optimizer'].param_groups):
                writer.add_scalar(f'lr/learning_rate_{i}', x['lr'], global_state['global_step'])
