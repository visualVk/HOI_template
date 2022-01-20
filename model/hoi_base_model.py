import math
import sys
from typing import List
from model.base_model import TrainBaseModel
from torch.utils.data import Dataset, DataLoader
from loss.hoi_criterion import SetCriterion
from utils.hoi_matcher import build_matcher as build_hoi_matcher
import utils.misc as utils
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
import easydict


class HoiTrainModel(TrainBaseModel):

    def __init__(self, args: argparse.Namespace, config: easydict.EasyDict,
                 dataset: Dataset, models: List[nn.Module], device=torch.device('cpu'), p=0.6):
        super(HoiTrainModel, self).__init__(
            args, config, dataset, models, device, p)

    def init_criterion(self):
        assert self.config.DATASET.NAME in [
            'hico', 'vcoco', 'hoia'], self.config.DATASET.NAME
        if self.config.DATASET.NAME in ['hico']:
            num_classes = 91
            num_actions = 118
        elif self.config.DATASET.NAME in ['vcoco']:
            num_classes = 91
            num_actions = 30
        else:
            num_classes = 12
            num_actions = 11

        matcher = build_hoi_matcher(self.config)

        weight_dict = dict(loss_ce=1, loss_bbox=self.config.CRITERION.BBOX_LOSS_COEF,
                           loss_giou=self.config.CRITERION.GIOU_LOSS_COEF)

        # TODO this is a hack
        if self.config.MODEL.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(self.config.MODEL.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']

        criterion = SetCriterion(num_classes=num_classes, num_actions=num_actions, matcher=matcher,
                                 weight_dict=weight_dict, eos_coef=self.config.CRITERION.EOS_COEF, losses=losses)
        criterion.to(self.device)
        self.criterion = [criterion]

    def init_optimizer(self):
        param_dicts = [
            {"params": [p for n, p in self.model_modules[0].named_parameters(
            ) if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model_modules[0].named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.config.TRAIN.LR,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_dicts, lr=self.config.TRAIN.LR_BACKBONE, weight_decay=self.config.TRAIN.WD)

    def train_one_epoch(self, epoch):
        step = 0
        with tqdm(total=len(self.train_dataloader), leave=True, desc=f"train epoch {epoch}",
                  ncols=100, unit='it', unit_scale=True) as tbar:

            model = self.models[0]
            for i, (samples, targets) in enumerate(self.train_dataloader):
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items() if k not in [
                    'image_id']} for t in targets]

                outputs = model(samples)
                criterion = self.criterion[0]
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                losses = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict.keys() if k in weight_dict)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                              for k, v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                if self.config.TRAIN.CLIP_MAX_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.TRAIN.CLIP_MAX_NORM)
                losses.backward()
                self.optimizer.step()

                tbar.update()
                tbar.set_postfix(
                    loss=losses.detach().cpu().item(),
                    lr1=self.optimizer.param_groups[0]['lr'],
                    lr2=self.optimizer.param_groups[1]['lr'])
                self.train_meter.update(losses.detach().cpu().item())
                step += 1

    def val_one_epoch(self, epoch):
        pass
