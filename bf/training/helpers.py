import argparse
import datetime
import importlib
import logging
import os
import re
import sys

from bf.utils.config_wrapper import ConfigWrapper

import torch


def _get_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    if os.path.isdir(checkpoint_path):
        pattern = re.compile('^ckpt-([0-9]+).pt$')
        checkpoint = max([(x, int(pattern.match(x)[1])) for x in os.listdir(checkpoint_path) if pattern.match(x)],
                         key=lambda x: x[1], default=None)
        if checkpoint:
            return os.path.join(checkpoint_path, checkpoint[0])
    return None

def load_config(args):
    if not os.path.exists(args.config):
        logging.error(f'XX File does not exist {args.config}')
        sys.exit(1)

    logging.info(f'>> Loading configuration from {args.config}')
    config_spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)

    config = ConfigWrapper(config)
    config.set_phases(args.phases)

    return config

def init_checkpoint(args):
    checkpoint = _get_checkpoint(args.checkpoint) if args.checkpoint else None

    if checkpoint:
        logging.info(f'>> Restoring from {checkpoint}')
        state = torch.load(checkpoint)
        if 'model' in state:
            if args.load_weights:
                del state['model']
            elif 'model_dict' in state:
                del state['model_dict']
        torch.cuda.empty_cache()
    else:
        state = {}

    if state and os.path.isdir(args.checkpoint) and not args.new_checkpoint:
        checkpoint_dir = args.checkpoint
    else:
        checkpoint_dir = os.path.join(args.save_dir, f'{datetime.datetime.today():%F-%H%M%S}')

    logging.info(f'>> Checkpoints will be saved to {checkpoint_dir}')

    if not args.debug and 'train' in args.phases:
        init_file_logger(checkpoint_dir)

    return state, checkpoint_dir

def init_file_logger(log_dir):
    os.path.exists(log_dir) or os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'train.log')
    file_handler = logging.FileHandler(log_path)
    logging.getLogger().addHandler(file_handler)

def get_default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.py')
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--phases', nargs='+', default=['train', 'eval'])
    parser.add_argument('--video', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--new_checkpoint', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--load_weights', action='store_true', help='Restore from weigths rather than full model when loading from checkpoint')
    return parser

def get_csv_log_file(args, log_dir):
    if args.debug:
        return os.devnull
    else:
        return os.path.join(log_dir, 'log.csv')

def init_logger(args):
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
