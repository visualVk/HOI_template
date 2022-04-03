import argparse
from typing import Optional

import torch
import math
from easydict import EasyDict
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from config.upt_vcoco_config import config as Cfg

from model.base_model import Train
from utils.misc import AverageMeter
from utils.model import adapt_device


class UPT_Trainer(Train):

    def __init__(self,
                 model: nn.Module,
                 args: argparse.Namespace,
                 config: EasyDict,
                 is_train: bool = True,
                 device: torch.device = torch.device('cpu'),
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
            device,
            accuracy,
            criterion,
            optimizer,
            lr_scheduler,
            sampler)
        _, model = adapt_device(
            model,
            config.DDP,
            config.CUDNN.ENABLED,
            args.local_rank)
        self.model = model

    def one_epoch(
            self,
            dataloader,
            meter,
            writer: SummaryWriter,
            epoch,
            stage='Train'):
        with tqdm(total=len(dataloader), ncols=140, desc=f"{stage} {epoch}") as tbar:
            for i, (inputs, targets) in enumerate(dataloader):
                if stage == "Train":
                    loss = self.model(inputs, targets)
                    interaction_loss = loss["interaction_loss"].detach().item()
                    # print(f"\n{interaction_loss}")
                    if not math.isinf(interaction_loss) and not math.isnan(
                            interaction_loss):
                        meter.update(interaction_loss)

                    self.optimizer.zero_grad(set_to_none=True)
                    tot_loss = sum(l for _, l in loss.items())
                    with torch.autograd.detect_anomaly():
                        tot_loss.backward()
                    if Cfg.TRAIN.CLIP_MAX_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), Cfg.TRAIN.CLIP_MAX_NORM)
                    self.optimizer.step()

                    tbar.set_postfix(
                        loss=interaction_loss,
                        # pose_loss=loss["pose_loss"].detach().cpu().item()
                    )
                    tbar.update()
                else:
                    detections = self.model(inputs)

                    tbar.set_postfix(detections=detections)
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
        pass


def build_upt_engine(upt: nn.Module, config, args):
    for p in upt.detector.parameters():
        p.requires_grad = False
    # for n, p in upt.pose_net.named_parameters():
    #     rel_name = ["conv1", "conv1d", "layer1", "layer2", "layer3", "layer4", "bn1", "maxpool", "relu"]
    #     for rn in rel_name:
    #         if rn in n:
    #             p.requires_grad = False
    param_dicts = [{
        "params": [p for n, p in upt.named_parameters()
                   if "interaction_head" in n and p.requires_grad]
    }]

    optim = torch.optim.AdamW(
        param_dicts, lr=config.TRAIN.LR_HEAD,
        weight_decay=config.TRAIN.WD
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, config.TRAIN.LR_DROP)

    upt_trainer = UPT_Trainer(
        upt, args, config, device=torch.device(
            args.local_rank), lr_scheduler=lr_scheduler)
    upt_trainer.update_optimizer(optim)
    return upt_trainer
