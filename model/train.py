from abc import ABCMeta, abstractmethod
from asyncio.log import logger
from typing import List
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from utils.misc import AverageMeter
from utils.draw_tensorboard import TensorWriter
from logging import Logger
import utils.misc as utils
import torch
import easydict
import argparse


class TrainMethod():
    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def eval_one_epoch(self, epoch):
        pass


class TestMethod(metaclas=ABCMeta):
    @abstractmethod
    def eval(self):
        pass


class Train(object):
    def __init__(self, model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 lr_scheduler: optim.lr_scheduler._LRScheduler,
                 dataloaders: List[DataLoader], sampler: Sampler,
                 config: easydict.EasyDict, args: argparse.Namespace):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.sampler = sampler

        self.config = config
        self.args = args

    def train(self, eval=True):
        begin_epoch = self.config.TRAIN.BEGIN_EPOCH
        end_epoch = self.config.TRAIN.END_EPOCH
        writer = TensorWriter().writer

        # ====== dataloader ======
        if len(self.dataloaders) == 2 and eval:
            logger.error('you choose eval mode, but have only one dataloader!')
            raise ValueError("train mode doesn't match with dataloader")
        train_dataloader, eval_dataloader = self.dataloaders[0], None
        if eval:
            eval_dataloader = self.dataloaders[1]

        # ====== main thread ======
        ddp = self.config.DDP
        is_main = False
        if ddp and utils.is_main_process():
            is_main = True
        elif not ddp:
            is_main = True

        # ====== begin epoch ======
        for epoch in range(begin_epoch, end_epoch):

            # ===== train =====
            self.model.train()
            train_meter = AverageMeter('train meter')
            if ddp:
                train_dataloader.sampler.set_epoch(epoch)

            self.train_one_epoch(train_dataloader, train_meter, epoch)
            train_meter.synchronize_between_process()

            if is_main:
                writer.add_scalar("train global loss",
                                  train_meter.global_avg(), epoch)
                # writer.add_scalar(f"train local_rank-{self.args.local_rank} loss", train_meter.avg())

            # ===== eval =====
            if eval:
                self.model.eval()
                eval_meter = AverageMeter('eval meter')

                if ddp:
                    eval_dataloader.sampler.set_epoch(epoch)

                self.eval_one_epoch(eval_dataloader, eval_meter, epoch)
                eval_meter.synchronize_between_process()

                if is_main:
                    writer.add_scalar("eval global loss",
                                      eval_meter.global_avg(), epoch)

    @abstractmethod
    def train_one_epoch(self, dataloader: DataLoader, meter: AverageMeter, writer: TensorWriter, epoch: int):
        pass

    @abstractmethod
    def eval_one_epoch(self, dataloader: DataLoader, meter: AverageMeter, writer: TensorWriter, epoch: int):
        pass
