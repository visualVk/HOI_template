import os
import warnings
from typing import OrderedDict, Optional, Union, Dict, List

import torch
import argparse
import os.path as osp
from logging import Logger
from config.config import config as Cfg
from torch import nn, Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from utils.logger import log_every_n
from utils.misc import NestedTensor

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
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
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

    model_without_ddp = model.to(device)
    model = with_ddp(model_without_ddp, local_rank) if ddp else with_dp(model, cuda)
    return model_without_ddp, model


def move_to_device(data: Union[Tensor,
                               List[Tensor],
                               Dict[str,
                                    Tensor],
                               NestedTensor],
                   device: Union[torch.device, str] = torch.device('cpu')):
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(data, (Tensor, NestedTensor)):
        data = data.to(device)
    elif isinstance(data, list):
        for i, d in enumerate(data):
            if isinstance(d, Tensor):
                assert isinstance(d, Tensor), "items in data must be Tensor"
                data[i] = d.to(device)
            elif isinstance(d, dict):
                for k, v in d.items():
                    assert isinstance(v, Tensor), "value must be Tensor!"
                    data[i][k] = v.to(device)
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


# def load_checkpoint(
#         model: nn.Module,
#         optimizer: torch.optim.Optimizer,
#         begin_epoch: int = 1,
#         lr_scheduler: Optional[nn.Module] = None):
#     root_path = os.path.join('./data/checkpoint')
#     checkpoint_filename = f"checkpoint_{begin_epoch - 1}.pth"
#     checkpoint_path = os.path.join(root_path, checkpoint_filename)
#     result = []
#     with open(checkpoint_path, 'r') as fp:
#         checkpoint = torch.load(fp, map_location='cpu')
#
#         _reload_parameters(checkpoint, model, 'model')
#         _reload_parameters(checkpoint, optimizer, 'optimizer')
#         result.append(model)
#         result.append(optimizer)
#         if lr_scheduler is not None:
#             _reload_parameters(checkpoint, lr_scheduler, 'lr_scheduler')
#             result.append(lr_scheduler)
#     return result
#
#
# def load_pretrained_model(model: nn.Module, filename: str):
#     pretrained_model_path = os.path.join('./data/', f'{filename}.pth')
#     with open(pretrained_model_path, 'rb') as fp:
#         pretrained_parameters = torch.load(fp, map_location='cpu')
#         _reload_parameters(pretrained_parameters, model, 'model')
#
    # return model


def load_checkpoint(cfg, model, optimizer, lr_scheduler, device, module_name='model'):
    last_iter = -1
    resume_path = cfg.MODEL.RESUME_PATH
    resume = cfg.TRAIN.RESUME
    if resume_path and resume:
        if osp.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location='cpu')
            # resume
            if 'model_state_dict' in checkpoint:
                model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
                log_every_n(20, f'==> model pretrained from {resume_path} \n')
            elif 'model' in checkpoint:
                if module_name == 'detr':
                    model.module.detr_head.load_state_dict(checkpoint['model'], strict=False)
                    log_every_n(20, f'==> detr pretrained from {resume_path} \n')
                else:
                    model.module.load_state_dict(checkpoint['model'], strict=False)
                    log_every_n(20, f'==> model pretrained from {resume_path} \n')
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log_every_n(20, f'==> optimizer resumed, continue training')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if 'optimizer_state_dict' in checkpoint and 'lr_state_dict' in checkpoint and 'epoch' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
                last_iter = checkpoint['epoch']
                log_every_n(20, f'==> last_epoch = {last_iter}')
            if 'epoch' in checkpoint:
                last_iter = checkpoint['epoch']
                log_every_n(20, f'==> last_epoch = {last_iter}')
            # pre-train
        else:
            log_every_n(40, f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    else:
        log_every_n(20, "==> train model without resume")

    return model, optimizer, lr_scheduler, last_iter