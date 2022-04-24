import argparse
from typing import Optional

import torch
from easydict import EasyDict
from torch import nn, optim
from torch.utils.data import Sampler, DataLoader

from engine.engine_no_tqdm import Engine
from utils import misc, relocate


class TRPoseEngine(Engine):
    def __init__(self,
                 model: nn.Module,
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
        super().__init__(
            model,
            args,
            config,
            is_train,
            use_amp,
            device,
            accuracy,
            criterion,
            optimizer,
            lr_scheduler,
            sampler)

    def _train_one_epoch(
            self,
            dataloader: DataLoader):
        max_epoch = self.config.TRAIN.END_EPOCH
        print_freq = self.config.PRINT_FREQ
        metric_logger = misc.MetricLogger(delimiter="  ")
        space_fmt = str(len(str(max_epoch)))
        header = 'Train Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(
            start_epoch=self._epoch + 1, end_epoch=max_epoch, fill=space_fmt)
        for inputs, targets in metric_logger.log_every(
                dataloader, print_freq, header):
            inputs = relocate.relocate_to_device(
                inputs, device=self.device)
            targets = relocate.relocate_to_device(
                targets, device=self.device)
            loss_dict = self._iterate_each_train_epoch(inputs, targets)
            metric_logger.update(tot_loss=self.loss, **loss_dict)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        metric_logger.synchronize_between_processes()
        self.writer.add_scalar(
            "global avg of total loss",
            metric_logger.meters["tot_loss"].global_avg,
            self._epoch)
        self.writer.add_scalar(
            "total loss in local",
            metric_logger.meters["tot_loss"].avg,
            self._epoch)
        # tot_loss = self.loss.detach().cpu().item()

    def _iterate_each_train_epoch(self, inputs, targets):
        outputs = self.model(inputs, targets)

        self.optimizer.zero_grad(set_to_none=True)

        tot_loss = self.criterion(outputs, targets["keypoints"])
        acc = self.accuracy(outputs, targets["keypoints"])
        self.loss = tot_loss

        tot_loss.backward()

        if self.config.TRAIN.CLIP_MAX_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.TRAIN.CLIP_MAX_NORM)

        self.optimizer.step()

        return dict(tot_loss=tot_loss, acc=acc)


def build_tr_pose_engine(model, criterion, accuracy, config, args):
    for p in model.detector.parameters():
        p.requires_grad = False

    param_dicts = [{
        "params": [p for n, p in model.named_parameters()
                   if "interaction_head" in n and p.requires_grad]
    }]

    optim = torch.optim.AdamW(
        param_dicts, lr=config.TRAIN.LR_HEAD,
        weight_decay=config.TRAIN.WD
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, config.TRAIN.LR_DROP)

    tr_pose_engine = TRPoseEngine(
        model,
        args,
        config,
        accuracy=accuracy,
        criterion=criterion,
        optimizer=optim,
        lr_scheduler=lr_scheduler)

    return tr_pose_engine
