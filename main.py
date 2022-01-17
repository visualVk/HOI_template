import argparse
import os
import sys
from config.config import config, update_config_by_yaml
from tools.train import train
from tools.test import test
import utils.misc as utils

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
    args.local_rank = int(os.environ["LOCAL_RANK"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "HOI Transformer training script", parents=[get_parse_args()])
    args = parser.parse_args()
    preprocess_config(args)
    utils.init_distributed_mode(args)
    utils.fix_random_seed()
    train(args, config)
    test(args,config)
