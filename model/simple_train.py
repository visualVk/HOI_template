import argparse
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import Sampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.model import move_to_device

from config.config import config as Cfg
from utils.misc import AverageMeter
from engine.engine import Engine


class SimpleTrain(Engine):
    def __init__(self,
                 model: nn.Module,
                 args: argparse.Namespace,
                 config: Cfg,
                 is_train: bool = True,
                 device: torch.device = torch.device('cpu'),
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 lr_scheduler: Optional[nn.Module] = None,
                 sampler: Optional[Sampler] = None):
        super().__init__(
            model,
            args,
            config,
            is_train,
            device,
            criterion,
            optimizer,
            lr_scheduler,
            sampler)

    def _train_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        with tqdm(total=len(dataloader), ncols=140, desc=f"train {epoch}") as tbar:
            for i, (images, targets) in enumerate(dataloader):
                # print(images.shape)
                images = move_to_device(images, self.device)
                targets = move_to_device(targets, self.device)

                x = self.model(images)

                self.optimizer.zero_grad()
                loss = self.criterion(
                    x, torch.ones_like(
                        x, dtype=torch.float, device=x.device))
                loss.backward()
                self.optimizer.step()

                tbar.set_postfix(
                    loss=loss.detach().cpu().item(),
                    lr=self.optimizer.param_groups[0]['lr'])
                tbar.update()

    def _eval_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        pass
