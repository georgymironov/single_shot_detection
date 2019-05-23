import argparse
import datetime
import importlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import sys

from copy import copy

import torch

from bf.utils.config_wrapper import ConfigWrapper


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

    if not args.debug:
        os.path.exists(checkpoint_dir) or os.makedirs(checkpoint_dir)
        logging.info(f'>> Checkpoints will be saved to {checkpoint_dir}')

        new_config_path = os.path.join(checkpoint_dir, 'config.py')
        if os.path.exists(args.config):
            if not os.path.exists(new_config_path) or not os.path.samefile(args.config, new_config_path):
                shutil.copy(args.config, new_config_path)

    return state, checkpoint_dir

def get_default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.py',
                        help='Path to a config file')
    parser.add_argument('--save-dir', type=str, default='./experiments',
                        help='Folder where checkpoints are going to be saved')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to restore checkpoint from. Overrides `save_dir`')
    parser.add_argument('--video', type=str,
                        help='Video or a folder (which will be searched recursively) for `test` phase')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug mode. Disables saving checkpoints and logs')
    parser.add_argument('--new-checkpoint', default=False, action='store_true',
                        help='Force checkpoint to be stores to `save_dir`')
    parser.add_argument('--tensorboard', default=False, action='store_true',
                        help='Log to tensorboard')
    parser.add_argument('--load-weights', default=False, action='store_true',
                        help='Restore from weigths rather than full model when loading from checkpoint')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='Run model on CPU')

    distributed = parser.add_argument_group('distributed')
    distributed.add_argument('--distributed', default=False, action='store_true',
                             help='Enable distributed training (only [single node - multi gpu] supported right now')
    distributed.add_argument('--nproc', type=int, default=torch.cuda.device_count(),
                             help='Number of jobs for distributed training (defaults to number of GPUs)')
    distributed.add_argument('--rank', type=int, default=0,
                             help='Rank of the current process')
    distributed.add_argument('--master-port', type=int, default=44444,
                             help='Free local port for worker communication')
    return parser

def get_csv_log_file(args, log_dir):
    if args.debug:
        return os.devnull
    else:
        return os.path.join(log_dir, 'log.csv')

def launch(args, main):
    if args.distributed:
        mp.set_start_method('spawn')
        processes = []

        for rank in range(args.nproc):
            _args = copy(args)
            _args.rank = rank

            process = mp.Process(target=main, args=(_args,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    else:
        main(args)
