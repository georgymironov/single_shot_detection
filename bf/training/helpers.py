import importlib
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
        print(f'XX File does not exist {path}')
        sys.exit(1)

    print(f'>> Loading configuration from {path}')
    config_spec = importlib.util.spec_from_file_location('config', path)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)

    return config

def load_checkpoint(path):
    checkpoint = _get_checkpoint(path) if path else None

    state = {}
    if checkpoint:
        print(f'>> Restoring from {checkpoint}')
        state = torch.load(checkpoint)

    return state
