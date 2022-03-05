import argparse
import os
import sys

import torch

from utils import misc
from config.config import config, update_config_by_yaml
from dataset import build_dataset
from dataset.hicodet import nested_tensor_collate
from model import build_model
from model.detr_train import DetrTrain
from utils.dataloader import build_dataloader
from utils.model import adapt_device

# os.chdir(sys.path[0])


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
        misc.init_distributed_mode(args)

    misc.fix_random_seed(args, config)
    train_image_dir = config.DATASET.IMAGES_TRAIN

    model, criterion, postprocessors = build_model(
        'detr', config, args)
    model_without_ddp, model = adapt_device(
        model, config.DDP, config.CUDNN.ENABLED, args.local_rank)
    param_dicts = [{"params": [p for n,
                               p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
                   {"params": [p for n,
                               p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": 1e-5,
                    },
                   ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200)

    dataset = build_dataset(
        'hico_det',
        train_image_dir,
        config.DATASET.ANNO_TRAIN)
    train_dataloader, val_dataloader = build_dataloader(
        dataset,
        collate_fn=nested_tensor_collate,
        ddp=config.DDP,
        num_workers=config.WORKERS,
        batch_size=1)

    detr_train = DetrTrain(
        model_without_ddp, args, config, device=torch.device(
            args.local_rank), lr_scheduler=lr_scheduler)

    detr_train.update_optimizer(optimizer)
    detr_train.update_train_dataloader(train_dataloader)
    detr_train.update_val_dataloader(val_dataloader)
    detr_train.update_criterion(criterion)
    detr_train.train()
    # train(args, config)
    # test(args,config)
