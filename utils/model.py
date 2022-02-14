import os
import warnings
from typing import OrderedDict, Optional, Union, Dict, List

import torch
import argparse
from logging import Logger
from config.config import config as Cfg
from torch import nn, Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def to_ddp(
        model: nn.Module,
        config: Cfg,
        args: argparse.Namespace) -> nn.Module:
    warnings.warn(
        "to_dpp is deprecated, please using adapt_device instead of it")
    is_ddp = torch.cuda.is_available() \
        and config.CUDNN.ENABLED \
        and config.DDP \
        and torch.cuda.device_count() > 1

    device = torch.device(args.local_rank) if torch.cuda.is_available(
    ) and config.CUDNN.ENABLED else torch.device('cpu')

    model = model.to(device)

    if is_ddp:
        Logger.warning(
            f"doesn't use ddp, cuda:{config.CUDNN.ENABLED and torch.cuda.is_available()}," +
            " ddp:{config.DDP}, gpus:{torch.cuda.device_count()}")

    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)

    return model


def with_ddp(model: nn.Module, local_rank: int):
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model


def with_dp(model: nn.Module, cuda: bool):
    if cuda:
        model = nn.DataParallel(model)
    return model


def adapt_device(
        model: nn.Module,
        ddp: bool,
        cuda: bool,
        local_rank: int) -> object:
    cuda = cuda and torch.cuda.is_available()
    ddp = ddp and cuda
    device = torch.device(local_rank if cuda else 'cpu')

    model = model.to(device)
    model = with_ddp(model, local_rank) if ddp else with_dp(model, cuda)
    return model


def move_to_device(data: Union[Tensor,
                               List[Tensor],
                               Dict[str,
                                    Tensor]],
                   device: torch.device = torch.device('cpu')):
    if isinstance(data, Tensor):
        data = data.to(device)
    elif isinstance(data, list):
        for d in data:
            assert isinstance(d, Tensor), "items in data must be Tensor"
            d = d.to(device)
    elif isinstance(data, dict):
        for k, v in data.items():
            assert isinstance(v, Tensor), "value must be Tensor!"
            data[k] = v.to(device)
    return data


def _reload_parameters(state_dict: OrderedDict,
                       attr: Union[nn.Module,
                                   torch.optim.Optimizer],
                       key: str):
    assert key in state_dict.keys(), \
        f"{key} is not in state_dict({state_dict.keys()})"
    attr.load_state_dict(state_dict[key])


def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        begin_epoch: int = 1,
        lr_scheduler: Optional[nn.Module] = None):
    root_path = os.path.join('./data/checkpoint')
    checkpoint_filename = f"checkpoint_{begin_epoch - 1}.pth"
    checkpoint_path = os.path.join(root_path, checkpoint_filename)
    result = []
    with open(checkpoint_path, 'r') as fp:
        checkpoint = torch.load(fp, map_location='cpu')

        _reload_parameters(checkpoint, model, 'model')
        _reload_parameters(checkpoint, optimizer, 'optimizer')
        result.append(model)
        result.append(optimizer)
        if lr_scheduler is not None:
            _reload_parameters(checkpoint, lr_scheduler, 'lr_scheduler')
            result.append(lr_scheduler)
    return result


def load_pretrained_model(model: nn.Module, filename: str):
    pretrained_model_path = os.path.join('./data/', f'{filename}.pth')
    with open(pretrained_model_path, 'r') as fp:
        pretrained_parameters = torch.load(fp, map_location='cpu')
        _reload_parameters(pretrained_parameters, model, 'model')

    return model
