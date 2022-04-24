import argparse
import copy
import gc
import math
import os
import pickle

import engine
from utils.vsrl_eval import VCOCOeval
import utils.vcoco_cached_helper as vhelper
from utils import misc
from typing import Optional

import torch
from engine.engine_no_tqdm import Engine
from easydict import EasyDict
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

# import engine
from utils import relocate, misc
from utils.vcoco_cached_helper import CacheTemplate


# class UPT_Trainer(engine.Engine):
#
#     def __init__(self,
#                  model: nn.Module,
#                  args: argparse.Namespace,
#                  config: EasyDict,
#                  is_train: bool = True,
#                  device: torch.device = torch.device('cpu'),
#                  accuracy: Optional[nn.Module] = None,
#                  criterion: Optional[nn.Module] = None,
#                  optimizer: Optional[optim.Optimizer] = None,
#                  lr_scheduler: Optional[object] = None,
#                  sampler: Optional[Sampler] = None):
#         super().__init__(
#             model,
#             args,
#             config,
#             is_train,
#             True,
#             device,
#             accuracy,
#             criterion,
#             optimizer,
#             lr_scheduler,
#             sampler)
#
#     def _train_one_epoch(
#             self,
#             dataloader: DataLoader):
#         if misc.is_main_process():
#             with tqdm(total=len(dataloader), ncols=140, desc=f"train {self._epoch}") as tbar:
#                 for i, (inputs, targets) in enumerate(dataloader):
#                     inputs = relocate.relocate_to_device(
#                         inputs, device=self.device)
#                     targets = relocate.relocate_to_device(
#                         targets, device=self.device)
#                     self._iterate_each_train_epoch(inputs, targets)
#                     # tot_loss = self.loss.detach().cpu().item()
#                     tbar.set_postfix(
#                         total_loss=self.loss.detach().item(),
#                         lr=self.optimizer.param_groups[0]["lr"])
#                     tbar.update()
#         else:
#             for i, (inputs, targets) in enumerate(dataloader):
#                 inputs = relocate.relocate_to_device(
#                     inputs, device=self.device)
#                 targets = relocate.relocate_to_device(
#                     targets, device=self.device)
#                 self._iterate_each_train_epoch(inputs, targets)
#
#     def _iterate_each_train_epoch(self, inputs, targets):
#         loss = self.model(inputs, targets)
#
#         self.optimizer.zero_grad(set_to_none=True)
#
#         tot_loss = sum(l for _, l in loss.items())
#         tot_loss = tot_loss.to(torch.float32)
#         self.loss = tot_loss
#         if torch.isnan(tot_loss):
#             self.train_meter.update(0)
#         else:
#             self.train_meter.update(tot_loss.detach().item())
#
#         tot_loss.backward()
#
#         if self.config.TRAIN.CLIP_MAX_NORM > 0:
#             torch.nn.utils.clip_grad_norm_(
#                 self.model.parameters(), self.config.TRAIN.CLIP_MAX_NORM)
#
#         self.optimizer.step()
#
#     @torch.no_grad()
#     def cache_vcoco(self, epoch, cache_dir='vcoco_cache', cache_name=''):
#         net = self.model
#         net.eval()
#
#         dataloader = self.val_dataloader
#         dataset = dataloader.dataset.dataset
#         all_results = []
#         for i, batch in enumerate(
#                 tqdm(dataloader, desc="cache vcoco", ncols=120)):
#             inputs = relocate.relocate_to_cuda(batch[0])
#             output = net(inputs)
#
#             # Skip images without detections
#             if output is None or len(output) == 0:
#                 continue
#             # Batch size is fixed as 1 for inference
#             assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
#             output = relocate.relocate_to_cpu(output[0], ignore=True)
#             # NOTE Index i is the intra-index amongst images excluding those
#             # without ground truth box pairs
#             image_id = dataset.image_id(i)
#             # Format detections
#             boxes = output['boxes']
#             boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
#             scores = output['scores']
#             actions = output['labels']
#             # Rescale the boxes to original image size
#             ow, oh = dataset.image_size(i)
#             h, w = output['size']
#             scale_fct = torch.as_tensor([
#                 ow / w, oh / h, ow / w, oh / h
#             ]).unsqueeze(0)
#             boxes_h *= scale_fct
#             boxes_o *= scale_fct
#
#             for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
#                 a_name = dataset.actions[a].split()
#                 result = CacheTemplate(
#                     image_id=image_id, person_box=bh.tolist())
#                 result[a_name[0] + '_agent'] = s.item()
#                 result['_'.join(a_name)] = bo.tolist() + [s.item()]
#                 all_results.append(result)
#
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
#         with open(os.path.join(cache_dir, f'cache_{cache_name}{epoch}.pkl'), 'wb') as f:
#             # Use protocol 2 for compatibility with Python2
#             pickle.dump(all_results, f, 2)
#             print(f"saved vcoco cached of epoch {epoch}")


class UPT_Trainer(Engine):

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
            True,
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
        loss = self.model(inputs, targets)

        self.optimizer.zero_grad(set_to_none=True)

        tot_loss = sum(l for _, l in loss.items())
        tot_loss = tot_loss.to(torch.float32)
        self.loss = tot_loss

        tot_loss.backward()

        if self.config.TRAIN.CLIP_MAX_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.TRAIN.CLIP_MAX_NORM)

        self.optimizer.step()

        return loss

    def eval(self):
        dataloader = self.val_dataloader
        max_epoch = self.config.TEST.END_EPOCH
        print_freq = self.config.PRINT_FREQ
        metric_logger = misc.MetricLogger(delimiter="  ")
        space_fmt = str(len(str(max_epoch)))
        header = 'Eval Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(
            start_epoch=self._epoch + 1, end_epoch=max_epoch, fill=space_fmt)
        self.model.eval()
        for inputs, targets in metric_logger.log_every(
                dataloader, print_freq, header):
            inputs = relocate.relocate_to_device(
                inputs, device=self.device)
            outputs = self.model(inputs)
            outputs = [output.to(torch.long) for output in outputs]

    def eval_pose(self):
        dataloader = self.val_dataloader
        header = 'Eval'
        print_freq = self.config.PRINT_FREQ
        metric_logger = misc.MetricLogger(delimiter="  ")
        self.model.eval()
        step = 0
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            filename = dataloader.dataset.dataset.filename(step)
            inputs = relocate.relocate_to_device(
                inputs, device=self.device)
            outputs = self.model(inputs)
            images = []
            for i, image in enumerate(inputs):
                image = image.detach().cpu()
                h, w = image.shape[1:]
                for keypoints in outputs[i]:
                    if keypoints.size(0) == 0: continue
                    for joint in keypoints:
                        sx = int(max(0, joint[0].item() - 5))
                        sx = sx if sx <= h else h - 5
                        sy = int(max(0, joint[1].item() - 5))
                        sy = sy if sy <= w else w - 5
                        ex = int(min(h, joint[0].item() + 5))
                        ey = int(min(w, joint[1].item() + 5))
                        image[:, sx:ex, sy:ey] = torch.tensor([255, 0, 0]).view(3, 1, 1).repeat(1, ex - sx, ey - sy)
                images.append(image)
            images = torch.stack(images)
            self.writer.add_images("keypoints image", images, step)
            step += 1

    @torch.no_grad()
    def cache_vcoco(self, epoch, cache_dir='vcoco_cache', cache_name=''):
        net = self.model
        net.eval()

        dataloader = self.val_dataloader
        dataset = dataloader.dataset.dataset
        all_results = []
        print_freq = self.config.PRINT_FREQ
        metric_logger = misc.MetricLogger(delimiter="  ")
        for i, batch in metric_logger.log_every(
                dataloader, print_freq, "Cache "):
            inputs = relocate.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = relocate.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(
                    image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, f'cache_{cache_name}{epoch}.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
            print(f"saved vcoco cached of epoch {epoch}")


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

    if config.TRAIN.RESUME:
        checkpoint_file = os.path.join(
            config.TRAIN.CHECKPOINT,
            f"checkpoint_{config.TRAIN.BEGIN_EPOCH}.pth")
        states_dict = torch.load(checkpoint_file, map_location="cpu")
        upt.load_state_dict(states_dict["model_state_dict"])
        optim.load_state_dict(states_dict["optimizer_state_dict"])
        lr_scheduler.load_state_dict(states_dict["lr_state_dict"])
        # lr_scheduler.step(config.TRAIN.BEGIN_EPOCH)

    upt_trainer = UPT_Trainer(
        upt,
        args,
        config,
        device=torch.device(
            args.local_rank),
        lr_scheduler=lr_scheduler)
    upt_trainer.update_optimizer(optim)
    return upt_trainer
