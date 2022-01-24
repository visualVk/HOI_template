from asyncio.log import logger
import os
import torch
import easydict
import argparse
from logging import Logger
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, DistributedSampler
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def to_ddp(model: nn.Module, config: easydict.EasyDict, args: argparse.Namespace) -> nn.Module:
    is_ddp = torch.cuda.is_available() \
        and config.CUDNN.ENABLED \
        and config.DDP \
        and torch.cuda.device_count() > 1

    device = torch.device(args.local_rank) if torch.cuda.is_available(
    ) and config.CUDNN.ENABLED else torch.device('cpu')

    model = model.to(device)

    if is_ddp:
        logger.warn(f"doesn't use ddp, cuda:{config.CUDNN.ENABLED and torch.cuda.is_available()},"
                    + " ddp:{config.DDP}, gpus:{torch.cuda.device_count()}")

    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)

    return model


def load_checkpoint(config: easydict.EasyDict, model: nn.Module, optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    eval=False):
    root_path = os.path.join('./')
    if eval:
        checkpoint_filename = config.MODEL.BEST_MODEL
    else:
        checkpoint_filename = f"checkpoint_{config.TRAIN.BEGIN_EPOCH - 1}.pth"

    checkpoint_path = os.path.join(root_path, checkpoint_filename)
    with open(checkpoint_path, 'r') as fp:
        checkpoint = torch.load(fp, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict['lr_scheduler']
    return model, optimizer, lr_scheduler


def split_train_dataset(dataset: Dataset, p=0.6):
    if p == 1:
        return dataset

    n = len(dataset)
    train_len = int(n * p)
    eval_len = n - train_len
    train_dataset, eval_dataset = random_split(dataset, [train_len, eval_len])
    return train_dataset, eval_dataset


def dataset_to_dataloader(dataset: Dataset, batch_size: int, num_workers: int, collate_fn, ddp=True):
    if ddp:
        sampler = RandomSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size, False,
                            sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    return dataloader
