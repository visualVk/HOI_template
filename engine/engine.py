import argparse
import copy
import gc
import math
import os.path
import warnings
from abc import abstractmethod
from typing import Any, Optional, List, Union
from utils.logger import log_every_n

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from easydict import EasyDict
from tqdm import tqdm

import utils.misc as utils
# from config.config import config as Cfg
from dataset.base import DataDict
from utils import relocate
from utils.draw_tensorboard import TensorWriter
from utils.misc import AverageMeter


class State:
    """
    Dict-based state class
    """

    def __init__(self) -> None:
        self._state = DataDict()

    def state_dict(self) -> dict:
        """Return the state dict"""
        return self._state.copy()

    def load_state_dict(self, dict_in: dict) -> None:
        """Load state from external dict"""
        for k in self._state:
            self._state[k] = dict_in[k]

    def fetch_state_key(self, key: str) -> Any:
        """Return a specific key"""
        if key in self._state:
            return self._state[key]
        else:
            raise KeyError("Inexistent key {}".format(key))

    def update_state_key(self, **kwargs) -> None:
        """Override specific keys in the state"""
        for k in kwargs:
            if k in self._state:
                self._state[k] = kwargs[k]
            else:
                raise KeyError("Inexistent key {}".format(k))


class Engine(object):
    def __init__(self, model: nn.Module,
                 args: argparse.Namespace,
                 config: EasyDict,
                 is_train: bool = True,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None,
                 accuracy: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 lr_scheduler: Optional[object] = None,
                 sampler: Optional[Sampler] = None):
        self.eval_meter = AverageMeter("evaluation total loss meter")
        self.train_meter = AverageMeter("train total loss meter")
        self.writer = TensorWriter().writer if args.local_rank == 0 else None
        self.accuracy = 0
        self._epoch = 0
        self.is_train = is_train
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.use_amp = use_amp
        self.sampler = sampler
        self.val_dataloader = None
        self.train_dataloader = None
        self.device = device if device is not None else torch.device(
            args.local_rank)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # torch.cuda.set_device(self.device)

        # log_every_n(20, f"==>rank{args.local_rank}-{self.device}")

        model = model.to(self.device)
        model = DDP(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)
        self.model = model
        self.loss = torch.tensor([0], device=self.device)
        self.acc = torch.tensor([0], device=self.device)

        self.accuracy = criterion if accuracy is None else accuracy
        self.config = config
        self.args = args

    def train(self, evaluate=False):
        begin_epoch = self.config.TRAIN.BEGIN_EPOCH
        end_epoch = self.config.TRAIN.END_EPOCH

        assert self.val_dataloader is not None \
            and self.train_dataloader is not None, \
            "you should load train dataloader, val dataloader or both by using update_dataloader"

        # ====== begin epoch ======
        for epoch in range(begin_epoch, end_epoch):
            self._epoch = epoch
            # ===== train =====
            self._train_one_epoch_before()
            self._train_one_epoch(self.train_dataloader)
            self._train_one_epoch_after()

            # ===== evaluate =====
            if evaluate and self.val_dataloader is not None:
                with torch.no_grad():
                    self._eval_one_epoch_before()
                    self._eval_one_epoch(self.val_dataloader)
                    self._eval_one_epoch_after()

    def save_checkpoint_file(self):
        checkpoint_dir = self.checkpoint_dir
        checkpoint_file = os.path.join(
            checkpoint_dir, checkpoint_name(self._epoch))
        states_dict = self._get_state_dict()
        torch.save(states_dict, checkpoint_file)

    def save_best_model(self):
        pass
        best_model_dir = self.model_dir
        best_model_file = os.path.join(
            best_model_dir, f"{self.model_name}.pth")
        states_dict = self._get_state_dict()
        torch.save(states_dict, best_model_file)

    def _get_state_dict(self):
        states_dict = {}
        states_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        states_dict["model_state_dict"] = self.model.module.state_dict()
        states_dict["epoch"] = self._epoch
        if self.lr_scheduler is not None:
            states_dict["lr_state_dict"] = self.lr_scheduler.state_dict()
        return states_dict

    # def get_attr_state_dict(self, attr_name: str):
    #     value = self.__getattribute__(attr_name)
    #     assert isinstance(value, (nn.Module, optim.Optimizer))
    #     if attr_name == "model":
    #         value = value.module.state_dict()
    #     return value


    @property
    def checkpoint_dir(self):
        checkpoint_dir = self.config.TRAIN.CHECKPOINT
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return checkpoint_dir

    def _train_one_epoch_before(self):
        self.model.train()
        if self.criterion is not None:
            self.criterion.train()
        self.train_meter.reset()
        self.train_dataloader.sampler.set_epoch(self._epoch)

    def _train_one_epoch_after(self):
        self.train_meter.synchronize_between_process()
        if utils.is_main_process():
            self.save_checkpoint_file()
            log_every_n(
                20, f"saved checkpoint of {self._epoch} in rank{utils.get_rank()}", 5)
            self.writer.add_scalar(
                "train global loss",
                self.train_meter.global_avg(),
                self._epoch)
        self._lr_schedular_step()

    def _train_one_epoch(
            self,
            dataloader: DataLoader):
        if utils.is_main_process():
            with tqdm(total=len(dataloader), ncols=140, desc=f"train {self._epoch}") as tbar:
                for i, (inputs, targets, _) in enumerate(dataloader):
                    inputs = relocate.relocate_to_device(
                        inputs, device=self.device)
                    targets = relocate.relocate_to_device(
                        targets, device=self.device)
                    self._iterate_each_train_epoch(inputs, targets)
                    # tot_loss = self.loss.detach().cpu().item()
                    tbar.set_postfix(
                        total_loss=self.loss.detach().item(),
                        lr=self.optimizer.param_groups[0]["lr"])
                    tbar.update()
        else:
            for i, (inputs, targets, _) in enumerate(dataloader):
                inputs = relocate.relocate_to_device(
                    inputs, device=self.device)
                targets = relocate.relocate_to_device(
                    targets, device=self.device)
                self._iterate_each_train_epoch(inputs, targets)

    @abstractmethod
    def _iterate_each_train_epoch(self, inputs, targets):
        pass

    @abstractmethod
    def _iterate_each_eval_epoch(self, inputs, targets):
        pass

    def _eval_one_epoch(
            self,
            dataloader: DataLoader):
        if utils.is_main_process():
            with tqdm(total=len(dataloader), ncols=140, desc=f"eval {self._epoch}") as tbar:
                for i, (inputs, targets, _) in enumerate(dataloader):
                    inputs = relocate.relocate_to_device(
                        inputs, device=self.device)
                    targets = relocate.relocate_to_device(
                        targets, device=self.device)
                    self._iterate_each_eval_epoch(inputs, targets)
                    tbar.set_postfix(total_loss=self.acc.detach().cpu().item())
                    tbar.update()
        else:
            for i, (inputs, targets, _) in enumerate(dataloader):
                inputs = relocate.relocate_to_device(
                    inputs, device=self.device)
                targets = relocate.relocate_to_device(
                    targets, device=self.device)
                self._iterate_each_eval_epoch(inputs, targets)

    def _eval_one_epoch_before(self):
        self.model.eval()
        if self.criterion is not None:
            self.criterion.eval()
        self.eval_meter.reset()

    def _eval_one_epoch_after(self):
        if utils.is_main_process():
            self.eval_meter.synchronize_between_process()
            if self.accuracy > self.eval_meter.global_avg():
                print(f"saving the best model in {self._epoch}")
                self.save_best_model()

            self.writer.add_scalar("eval global loss",
                                   self.eval_meter.global_avg(), self._epoch)

    def update_attr(self, key: str, val: Any):
        assert key in self.__dict__.keys(), \
            f"{key} isn't expected, only {self.__dict__.keys()} will be expected"
        self.__setattr__(key, val)

    def update_train_dataloader(self, train_dataloader: DataLoader):
        self.update_attr('train_dataloader', train_dataloader)

    def update_val_dataloader(self, val_dataloader: DataLoader):
        self.update_attr('val_dataloader', val_dataloader)

    def update_optimizer(self, optimizer: optim.Optimizer):
        self.update_attr('optimizer', optimizer)

    def update_criterion(self, criterion: nn.Module):
        self.update_attr('criterion', criterion)

    def _lr_schedular_step(self):
        if self.lr_scheduler is None:
            return
        self.lr_scheduler.step()

    def reload_eval_model_in_epoch(
            self,
            model: nn.Module, ):
        torch.cuda.set_device(self.device)
        model = model.to(self.device)
        model = DDP(
            model,
            device_ids=[
                self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True)
        del self.model
        gc.collect()
        self.model = model


def checkpoint_name(epoch: int):
    """

    Args:
        epoch:

    Returns:

    """
    return f"checkpoint_{epoch}.pth"
