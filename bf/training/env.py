import functools
import logging
import os
import random

import numpy as np
import torch


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

def init_logger(args):
    logging.basicConfig(format='%(message)s')
    if args.rank == 0:
        level = logging.DEBUG if args.debug else logging.INFO
    else:
        level = logging.ERROR
    logging.getLogger().setLevel(level)

def init_file_logger(args, log_dir):
    if args.rank == 0 and not args.debug and 'train' in args.phases:
        os.path.exists(log_dir) or os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'train.log')
        file_handler = logging.FileHandler(log_path)
        logging.getLogger().addHandler(file_handler)

def init_random_state(args, cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

def set_device(args, cfg):
    use_cuda = cfg.use_gpu and torch.cuda.is_available()

    if args.distributed:
        assert use_cuda
        device = f'cuda:{args.rank}'
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=f'tcp://127.0.0.1:{args.master_port}',
                                             world_size=args.nproc,
                                             rank=args.rank)
        num_gpus = 1
    else:
        device = 'cuda:0' if use_cuda else 'cpu'
        num_gpus = torch.cuda.device_count()

    if use_cuda:
        cfg.update({'num_gpus': num_gpus})

    return device, use_cuda
