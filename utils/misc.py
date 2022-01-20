import torch.distributed as dist
import torch
import argparse
import os
import torch.nn as nn
import easydict
import numpy as np
import random
from model.nested_tensor import nested_tensor_from_tensor_list


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

    def synchronize_between_process(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]

    def __str__(self):
        fmt = "{}: val = {:.5f}, avg = {:.5f}".format(
            self.name, self.val, self._avg)
        return fmt


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


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


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_loss(loss: nn.Module, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad:
        reduce_loss = loss
        dist.all_reduce(reduce_loss)
        if average:
            reduce_loss /= world_size
    return reduce_loss


def fix_random_seed(args: argparse.Namespace, config: easydict):
    seed = config.SEED + args.local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
