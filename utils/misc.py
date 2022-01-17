import torch.distributed as dist
import torch
import argparse
import os
import numpy as np
import random


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0



def init_distributed_mode(args: argparse.Namespace):
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(
        backend="gloo", init_method="env://", rank=args.local_rank)


def fix_random_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
