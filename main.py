import argparse
import os
import sys

import torch

import utils.misc as utils
from config.config import config, update_config_by_yaml
from dataset import build_dataset
from loss.simple_criterion import SimpleCriterion
from model import build_model
from model.simple_train import SimpleTrain
from utils.dataloader import build_dataloader

os.chdir(sys.path[0])


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="HOI Transformer", add_help=False)
    parser.add_argument("--local_rank", default=0,
                        type=int, help="Local rank")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("--world_size", default=0, type=int, help="World size")
    parser.add_argument("-nr", "--nr", default=0, type=int,
                        help="ranking within the nodes")
    parser.add_argument(
        "--backend",
        default='gloo',
        type=str,
        help="ddp backend")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./config.yaml",
        help="config written by yaml path")

    return parser


def preprocess_config(args: argparse.Namespace):
    update_config_by_yaml(args.path)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if config.DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "HOI Transformer training script", parents=[get_parse_args()])
    args = parser.parse_args()
    preprocess_config(args)
    # whether to use DDP
    if config.DDP and config.CUDNN.ENABLED and torch.cuda.is_available():
        utils.init_distributed_mode(args)

    utils.fix_random_seed(args, config)
    model = build_model(
        "simple_net",
        config.CUDNN.ENABLED and torch.cuda.is_available(),
        config.DDP,
        args.local_rank)
    train_image_dir = os.path.join(
        config.DATASET.ROOT,
        config.DATASET.NAME,
        config.DATASET.TRAIN_IMAGES)
    dataset = build_dataset(
        'hico_det',
        train_image_dir,
        'D:\\code\\dpl\\data_an\\hico_train.json')
    train_dataloader, val_dataloader = build_dataloader(
        dataset,
        ddp=config.DDP,
        num_workers=config.WORKERS,
        batch_size=1)
    hoi_train = SimpleTrain(
        model, args, config, device=torch.device(
            args.local_rank))
    criterion = SimpleCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    hoi_train.update_optimizer(optimizer)
    hoi_train.update_train_dataloader(train_dataloader)
    hoi_train.update_val_dataloader(val_dataloader)
    hoi_train.update_criterion(criterion)
    hoi_train.train()
    # train(args, config)
    # test(args,config)
