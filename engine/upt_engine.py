import argparse
import copy
import gc
import math
import os
import pickle
from utils.vsrl_eval import VCOCOeval
import utils.vcoco_cached_helper as vhelper
from typing import Optional

import torch
from easydict import EasyDict
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.upt_vcoco_config import config as Cfg
from model.base_model import Engine
from utils import relocate, misc
from utils.misc import AverageMeter
from utils.model import adapt_device
from utils.vcoco_cached_helper import CacheTemplate


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
                    # if not math.isinf(interaction_loss) and not math.isnan(
                    #         interaction_loss):
                    #     meter.update(interaction_loss)

                    self.optimizer.zero_grad(set_to_none=True)
                    print(loss.items())
                    tot_loss = sum(l for _, l in loss.items())
                    meter.update(tot_loss.detach().item())
                    # with torch.autograd.detect_anomaly():
                    tot_loss.backward()
                    if Cfg.TRAIN.CLIP_MAX_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), Cfg.TRAIN.CLIP_MAX_NORM)
                    self.optimizer.step()

                    tbar.set_postfix(
                        loss=tot_loss,
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

    # def _eval_one_epoch_before(self, epoch: int, device="cuda"):
    #     checkpoint_filename = os.path.join(
    #         self.config.TRAIN.CHECKPOINT,
    #         f"checkpoint_{epoch}.pth")
    #     net_state_dict = torch.load(checkpoint_filename, map_location=device)
    #     self.model.load_state_dict(net_state_dict["model"])

    def _eval_one_epoch(
            self,
            dataloader: DataLoader,
            meter: AverageMeter,
            writer: SummaryWriter,
            epoch: int):
        # self.eval_vcoco(epoch, writer)
        self._cache_vcoco(epoch, "data/cache")

    def eval_vcoco(self, epoch, writer: SummaryWriter):
        # TODO: need to test, not support ddp
        vsrl_annot_file = "data/mscoco2014/vcoco_test.json"
        coco_file = "data/mscoco2014/instances_vcoco_all_2014.json"
        split_file = "data/mscoco2014/splits/vcoco_test.ids"

        # Change this line to match the path of your cached file
        det_file = f"./data/cache/cache_{epoch}.pkl"

        print(f"Loading cached results from {det_file}.")
        vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
        mAP_a, mAP_r_1, mAP_r_2 = vcocoeval._do_eval(det_file, ovr_thresh=0.5)
        # if misc.is_main_process():
        writer.add_scalar("mAP of agent", mAP_a, epoch)
        writer.add_scalar("mAP of role in scenario 1", mAP_r_1, epoch)
        writer.add_scalar("mAP of role in scenario 2", mAP_r_2, epoch)

    # @torch.no_grad()
    # def cache_vcoco(self, cache_dir="vcoco_cache"):
    #     begin_epoch = self.config.TEST.BEGIN_EPOCH
    #     end_epoch = self.config.TEST.END_EPOCH
    #     for epoch in range(begin_epoch, end_epoch):
    #         # self._eval_one_epoch_before(epoch, device="cpu")
    #         self._cache_vcoco(epoch, cache_dir)

    @torch.no_grad()
    def _cache_vcoco(self, epoch, cache_dir='vcoco_cache'):
        net = self.model
        net.eval()

        dataloader = self.val_dataloader
        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(
                tqdm(dataloader, desc="cache vcoco", ncols=120)):
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
        with open(os.path.join(cache_dir, f'cache_pose_{epoch}.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
            print(f"saved vcoco cached of epoch {epoch}")


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
