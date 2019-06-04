import functools
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
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

def set_random_state(args, cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

def set_device(args, cfg):
    use_cuda = not args.cpu and torch.cuda.is_available()

    if args.distributed:
        assert use_cuda
        device = f'cuda:{args.rank}'
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl',
                                init_method=f'tcp://127.0.0.1:{args.master_port}',
                                world_size=args.nproc,
                                rank=args.rank)
        num_gpus = 1
    elif use_cuda:
        device = 'cuda:0'
        num_gpus = torch.cuda.device_count()
    else:
        device = 'cpu'
        num_gpus = 0

    if use_cuda:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg.update({'num_gpus': num_gpus})

    return device, use_cuda
