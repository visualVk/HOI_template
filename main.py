import argparse
import numpy as np
import random
import os
import sys
import torch.nn as nn
from config.config import config, update_config_by_yaml, _update_dict
from tools.train import train
import utils.misc as utils
import torch
import torch.distributed as dist

os.chdir(sys.path[0])


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="HOI Transformer", add_help=False)
    parser.add_argument("--local_rank", default=-1,
                        type=int, help="Local rank")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("--world_size", default=0, type=int, help="World size")
    parser.add_argument("-nr", "--nr", default=0, type=int,
                        help="ranking within the nodes")
    parser.add_argument("-p", "--path", type=str,
                        default="./config.yaml", help="config written by yaml path")

    return parser


def preprocess_config(args: argparse.Namespace):
    update_config_by_yaml(args.path)
    args.local_rank = os.environ["LOCAL_RANK"]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "HOI Transformer training script", parents=[get_parse_args()])
    args = parser.parse_args()
    print(args)
    preprocess_config(args)
    init_distributed_mode(args)
    fix_random_seed()
    train(args, config)
