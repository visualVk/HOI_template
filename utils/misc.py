import torch.distributed as dist
import torch
import argparse
import os
import numpy as np
import random


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.sum = 0
        self._avg = 0
        self.count = 0

    def avg(self):
        return self._avg

    def __str__(self):
        fmt = "{}: val = {:.5f}, avg = {:.5f}".format(
            self.name, self.val, self._avg)
        return fmt


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
