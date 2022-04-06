import argparse
import copy
import gc
import math
import os.path
import warnings
from abc import abstractmethod
from typing import Any, Optional, List, Union

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict
from tqdm import tqdm

import utils.misc as utils
# from config.config import config as Cfg
from utils.draw_tensorboard import TensorWriter
from utils.misc import AverageMeter


class Engine(object):
    def __init__(self, model: nn.Module,
                 args: argparse.Namespace,
                 config: EasyDict,
                 is_train: bool = True,
                 device: torch.device = torch.device('cpu'),
                 accuracy: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 lr_scheduler: Optional[object] = None,
                 sampler: Optional[Sampler] = None):
        self.model = model
        self.is_train = is_train
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sampler = sampler
        self.val_dataloader = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.device = device
        self.accuracy = criterion if accuracy is None else accuracy
        self.config = config
        self.args = args

    def train(self, evaluate=False):
        begin_epoch = self.config.TRAIN.BEGIN_EPOCH
        end_epoch = self.config.TRAIN.END_EPOCH
        writer = TensorWriter().writer
        accuracy = math.inf

        assert self.val_dataloader is not None \
            and self.train_dataloader is not None, \
            "you should load train dataloader, val dataloader or both by using update_dataloader"
        # ====== main thread ======
        ddp = self.use_ddp
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
                self.train_dataloader.sampler.set_epoch(epoch)
            self._train_one_epoch_before()
            self._train_one_epoch(
                self.train_dataloader, train_meter, writer, epoch)
            self._train_one_epoch_after()
            train_meter.synchronize_between_process()

            if is_main:
                print(f"saving checkpoint of {epoch}")
                self.save_checkpoint_file(epoch)
                writer.add_scalar("train global loss",
                                  train_meter.global_avg(), epoch)
                # writer.add_scalar(f"train local_rank-{self.args.local_rank} loss", train_meter.avg())

            # ===== evaluate =====
            if evaluate and self.val_dataloader is not None:
                self.model.eval()
                eval_meter = AverageMeter('eval meter')

                if ddp:
                    self.val_dataloader.sampler.set_epoch(epoch)
                with torch.no_grad():
                    self._eval_one_epoch_before()
                    self._eval_one_epoch(
                        self.val_dataloader, eval_meter, writer, epoch)
                    self._eval_one_epoch_after()
                    eval_meter.synchronize_between_process()

                    if is_main:
                        if accuracy > eval_meter.global_avg():
                            print(f"saving the best model in {epoch}")
                            self.save_best_model()

                        writer.add_scalar("eval global loss",
                                          eval_meter.global_avg(), epoch)

    def eval(self):
        assert self.test_dataloader is not None, "in evaluate mode, test_dataloader shouldn't be None"
        begin_epoch = self.config.TEST.BEGIN_EPOCH
        end_epoch = self.config.TEST.END_EPOCH
        eval_meter = AverageMeter('eval_meter in evaluate mode')
        writer = TensorWriter().writer
        for epoch in tqdm(range(begin_epoch, end_epoch)):
            with torch.no_grad():
                self._eval_one_epoch_before(epoch)
                self.model.eval()
                self._eval_one_epoch(
                    self.test_dataloader, eval_meter, writer, epoch)
                self._eval_one_epoch_after()

    def save_checkpoint_file(self, epoch: int):
        checkpoint_dir = self.checkpoint_dir
        checkpoint_file = os.path.join(
            checkpoint_dir, checkpoint_name(epoch))

        attr_needed_saving = ['optimizer', 'model']
        states_dict = {}
        for attr_name in attr_needed_saving:
            state_dict = self.get_attr_state_dict(attr_name)
            states_dict[attr_name] = state_dict

        torch.save(states_dict, checkpoint_file)

    def save_best_model(self):
        best_model_dir = self.model_dir
        best_model_file = os.path.join(
            best_model_dir, f"{self.model_name}.pth")
        attr_needed_saving = ['optimizer', 'model']
        states_dict = {}
        for attr_name in attr_needed_saving:
            state_dict = self.get_attr_state_dict(attr_name)
            states_dict[attr_name] = state_dict

        torch.save(states_dict, best_model_file)

    def get_attr_state_dict(self, attr_name: str):
        value = self.__getattribute__(attr_name)
        assert isinstance(value, (nn.Module, optim.Optimizer))
        value = value.state_dict()
        return value

    def adapt_device(self, data: Union[Tensor, List[Tensor]]):
        warnings.warn(
            "adapt_device is deprecated and will be instead by utils.models.move_to_device")
        if self.use_cuda:
            if isinstance(data, Tensor):
                data = data.to(self.device)
            elif isinstance(data, list):
                for d in data:
                    d = d.to(self.device)
        return data

    @property
    def use_cuda(self):
        return self.config.CUDNN.ENABLED and torch.cuda.is_available()

    @property
    def use_ddp(self):
        return self.config.DDP and self.use_cuda

    @property
    def resume(self):
        return self.config.TRAIN.RESUME

    @property
    def checkpoint_dir(self):
        checkpoint_dir = self.config.TRAIN.CHECKPOINT
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return checkpoint_dir

    @property
    def model_name(self):
        return self.config.MODEL.NAME

    @property
    def model_dir(self):
        model_dir = self.config.MODEL.BEST_MODEL
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        return model_dir

    def _train_one_epoch_before(self):
        pass

    def _train_one_epoch_after(self):
        self._lr_schedular_step()

    @abstractmethod
    def _train_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        pass

    @abstractmethod
    def _eval_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        pass

    def _eval_one_epoch_before(self, epoch: int):
        pass

    def _eval_one_epoch_after(self):
        pass

    def update_attr(self, key: str, val: Any):
        assert key in self.__dict__.keys(), \
            f"{key} isn't expected, only {self.__dict__.keys()} will be expected"
        self.__setattr__(key, val)

    def update_train_dataloader(self, train_dataloader: DataLoader):
        self.update_attr('train_dataloader', train_dataloader)

    def update_val_dataloader(self, val_dataloader: DataLoader):
        self.update_attr('val_dataloader', val_dataloader)

    def update_test_dataloader(self, test_dataloader: DataLoader):
        self.update_attr('test_dataloader', test_dataloader)

    def update_optimizer(self, optimizer: optim.Optimizer):
        self.update_attr('optimizer', optimizer)

    def update_criterion(self, criterion: nn.Module):
        self.update_attr('criterion', criterion)

    def _lr_schedular_step(self):
        if self.lr_scheduler is None:
            return
        self.lr_scheduler.step()


def checkpoint_name(epoch: int):
    """

    Args:
        epoch:

    Returns:

    """
    return f"checkpoint_{epoch}.pth"
