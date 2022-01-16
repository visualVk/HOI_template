import argparse
import numpy as np
import random
import os
import sys
import torch.nn as nn
from config.config import config, update_config_by_yaml, _update_dict
from tools.train import train
import torch
import torch.distributed as dist

os.chdir(sys.path[0])


def args_parser():
    parser = argparse.ArgumentParser(description="HOI arguments")
    parser.add_argument("-lrk", "--local_rank", default=1,
                        type=int, help="number of local rank")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("-nr", "--nr", default=0, type=int,
                        help="ranking within the nodes")
    parser.add_argument("-p", "--path", type=str,
                        default="./config.yaml", help="config written by yaml path")
    args = parser.parse_args()
    update_config_by_yaml(args.path)

    if torch.cuda.device_count() < len(config.LOCAL_RANK):
        _update_dict(
            "LOCAL_RANK", config.LOCAL_RANK[:torch.cuda.device_count()])
    args.local_rank = config.LOCAL_RANK
    args.gpus = len(args.local_rank)

    return args


def init_distributed_mode(args: argparse.Namespace):
    world_size = args.gpus * args.nodes  # number of threads

    torch.cuda.set_device(*args.local_rank)

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=world_size, rank=0)


def fix_random_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = args_parser()
    fix_random_seed()
    init_distributed_mode(args)
    train()
