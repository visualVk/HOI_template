import argparse
import math
import sys
from typing import Optional
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from config import config as Cfg
from utils import misc
from utils.misc import AverageMeter
from utils.model import move_to_device
from model.base_model import Engine
from model.ds import nested_tensor_from_tensor_list


class DetrTrain(Engine):
    def one_epoch(self, dataloader, meter, writer, epoch, stage='Train'):
        with tqdm(total=len(dataloader), ncols=140, desc=f"{stage} {epoch}") as tbar:
            for i, (images, targets) in enumerate(dataloader):
                samples = move_to_device(images, 'cuda')
                targets = move_to_device(targets, 'cuda')

                outputs = self.model(samples)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict.keys() if k in weight_dict)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {
                    f'{k}_unscaled': v for k,
                    v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {
                    k: v * weight_dict[k] for k,
                    v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                if stage == 'Train':
                    if not math.isfinite(loss_value):
                        print(
                            "Loss is {}, stopping training".format(loss_value))
                        print(loss_dict_reduced)
                        sys.exit(1)

                    self.optimizer.zero_grad()
                    losses.backward()
                    if Cfg.TRAIN.CLIP_MAX_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), Cfg.TRAIN.CLIP_MAX_NORM)
                    self.optimizer.step()

                meter.update(loss_value)

                tbar.set_postfix(loss=loss_value)
                tbar.update()

    def _train_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        self.one_epoch(dataloader, meter, writer, epoch)

    def _eval_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        self.one_epoch(dataloader, meter, writer, epoch, 'Eval')
