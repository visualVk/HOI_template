import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.base_model import Train
from utils.misc import AverageMeter


class UPT_Trainer(Train):

    def _train_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        pass

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
