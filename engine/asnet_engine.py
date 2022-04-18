import argparse
import math
import sys
from typing import Optional, Union

import torch
from easydict import EasyDict
from torch import nn, optim
from torch.utils.data import Sampler, DataLoader
from tqdm import tqdm

from utils import misc, relocate
from utils.misc import write_dict_to_json
from .engine import Engine


class ASNetEngine(Engine):
    def __init__(self,
                 model: nn.Module,
                 args: argparse.Namespace,
                 config: EasyDict,
                 postprocessors,
                 is_train: bool = True,
                 device: Optional[Union[torch.device, str]] = None,
                 accuracy: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 lr_scheduler: Optional[object] = None,
                 sampler: Optional[Sampler] = None,
                 performance_indicator="mAP",
                 max_norm=0):
        super().__init__(
            model,
            args,
            config,
            is_train,
            True,
            device,
            accuracy,
            criterion,
            optimizer,
            lr_scheduler,
            sampler)
        self.performance_indicator = performance_indicator
        self.postprocessors = postprocessors
        self.max_norm = max_norm

    def _train_one_epoch(
            self,
            dataloader: DataLoader):
        if misc.is_main_process():
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

    def _iterate_each_train_epoch(self, inputs, targets):
        outputs = self.model(inputs)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k,
            v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        self.loss = losses_reduced_scaled

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        self.optimizer.zero_grad()
        losses.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_norm)
        self.optimizer.step()

        self.train_meter.update(loss_value)

    def evaluate(self, mode='hico', rel_topk=100):
        self.model.eval()
        eval_loader = self.val_dataloader
        results = []
        count = 0
        for data in tqdm(eval_loader):
            imgs, targets, filenames = data
            imgs = relocate.relocate_to_device(imgs, device=self.device)
            # targets are list type
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]
            bs = len(imgs)
            target_sizes = targets[0]['size'].expand(bs, 2)
            target_sizes = target_sizes.to(self.device)
            outputs_dict = self.model(imgs)
            file_name = filenames[0]
            pred_out = self.postprocessors(
                outputs_dict, file_name, target_sizes, rel_topk=rel_topk)
            results.append(pred_out)
            count += 1
        # save the result
        result_path = f'{self.config.OUTPUT_ROOT}/pred.json'
        write_dict_to_json(results, result_path)

        # eval
        if mode == 'hico':
            from utils.hico_eval import hico
            eval_tool = hico(
                annotation_file='data/hico/hico/test_hico.json',
                train_annotation='data/hico/hico/trainval_hico.json')
            mAP = eval_tool.evalution(results)

        return mAP
