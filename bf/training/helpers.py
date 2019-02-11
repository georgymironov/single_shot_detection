import datetime
import importlib
import logging
import os
import re
import sys

import torch


def _get_checkpoint(checkpoint_dir):
    pattern = re.compile('^ckpt-([0-9]+).pt$')
    checkpoint = max([(x, int(pattern.match(x)[1])) for x in os.listdir(checkpoint_dir) if pattern.match(x)],
                     key=lambda x: x[1], default=None)
    if checkpoint:
        return os.path.join(checkpoint_dir, checkpoint[0])
    return None

def load_config(path):
    if not os.path.exists(path):
        logging.error(f'XX File does not exist {path}')
        sys.exit(1)

    logging.info(f'>> Loading configuration from {path}')
    config_spec = importlib.util.spec_from_file_location('config', path)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)

    return config

def init_checkpoint(path, new_checkpoint=False, save_dir=None):
    checkpoint = _get_checkpoint(path) if path else None

    if checkpoint:
        logging.info(f'>> Restoring from {checkpoint}')
        state = torch.load(checkpoint)
    else:
        state = {}

    if state and not new_checkpoint:
        checkpoint_dir = path
    else:
        checkpoint_dir = os.path.join(save_dir, f'{datetime.datetime.today():%F-%H%M%S}')

    os.path.exists(checkpoint_dir) or os.makedirs(checkpoint_dir)
    logging.info(f'>> Checkpoints will be saved to {checkpoint_dir}')

    return state, checkpoint_dir

def init_file_logger(log_dir):
    os.path.exists(log_dir) or os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'train.log')
    file_handler = logging.FileHandler(log_path)
    logging.getLogger().addHandler(file_handler)
