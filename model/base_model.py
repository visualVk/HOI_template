from ast import Tuple
import torch.nn as nn
import argparse
import os
import torch
import easydict
from typing import List
import utils.misc as utils
from utils.misc import AverageMeter
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split, RandomSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


class TrainBaseModel(object):
    def __init__(self, args: argparse.Namespace, config: easydict,
                 dataset: Dataset, models: List[nn.Module], device=torch.device('cpu'), p=0.6):
        super().__init__()
        self.args = args
        self.config = config
        self.models = models
        self.dataset = dataset
        self.model_modules = models
        self.device = device
        self.train_meter = AverageMeter(
            f"{self.config.PROJECT_NAME} train tot loss")
        self.val_meter = AverageMeter(
            f"{self.config.PROJECT_NAME} val tot loss")
        self.p = p

        self.init_model()

    @classmethod
    def init_with_train_and_val(cls, args: argparse.Namespace, config: easydict,
                                train_dataset: Dataset, val_dataset: Dataset,
                                models: List[nn.Module], device=torch.device('cpu')):
        obj = cls(args, config, None, models, device)
        obj.train_dataset = train_dataset
        obj.val_dataset = val_dataset
        return obj

    def init_dataloader(self):
        # case 1: the simplest init method
        if self.dataset is not None:
            tot_len = len(self.dataset)
            train_len = int(tot_len * self.p)
            val_len = tot_len - train_len
            self.train_dataset, self.val_dataset = \
                random_split(self.dataset, [train_len, val_len])

        self.train_dataloader = self._init_dataloader(self.train_dataset)
        self.val_dataloader = self._init_dataloader(self.val_dataset)

    def _init_dataloader(self, dataset: Dataset) -> DataLoader:
        if self.config.DDP:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.config.TRAIN.BATCH_SIZE, sampler=sampler, shuffle=False,
                                num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
        return dataloader

    def init_model(self):
        if self.config.TRAIN.RESUME:
            checkpoint \
                = torch.load(os.path.join(self.config.TRAIN.CHECKPOINT,
                                          f"checkpoint_{self.config.TRAIN.BEGIN_EPOCH-1}.pth"),
                             map_location='cpu')
            for i, (m) in enumerate(self.models):
                m.load_state_dict(checkpoint[f"model_{i}"])

        if torch.cuda.is_available() and self.config.CUDNN.ENABLED:
            self.models = [m.to(self.device) for m in self.models]

            if torch.cuda.device_count() >= 1 and self.config.DDP:
                self.models = [DDP(m, device_ids=[self.args.local_rank],
                                   output_device=self.args.local_rank) for m in self.models]

        self.init_optimizer()
        self.init_scheduler()

        if self.config.TRAIN.RESUME:
            self.optimizer.load_state_dict(checkpoint['optim'])
            if self.lr_scheduler != None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("loaded checkpoint!")

        self.init_criterion()
        self.init_dataloader()

    def init_scheduler(self):
        self.lr_scheduler = None

    def train(self):
        # begin training
        begin_epoch = self.config.TRAIN.BEGIN_EPOCH
        end_epoch = self.config.TRAIN.END_EPOCH

        for epoch in range(begin_epoch, end_epoch):

            if self.config.DDP:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            self.train_meter.synchronize_between_process()
            self.train_meter.reset()

            if self.config.DDP:
                self.val_dataloader.sampler.set_epoch(epoch)
            self.val_one_epoch(epoch)
            self.val_meter.synchronize_between_process()
            self.val_meter.reset()

            if self.lr_scheduler != None:
                self.lr_scheduler.step()

            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        # save checkpoint
        if (self.config.DDP and utils.is_main_process()
                or not self.config.DDP) \
                and epoch >= self.config.TRAIN.SAVE_BEGIN \
                and epoch % self.config.TRAIN.INTERVAL_SAVE == 0:
            checkpoint_dict = {}

            for i in range(len(self.models)):
                model = self.model_modules[i]
                checkpoint_dict[f'model_{i}'] = model.state_dict()

            assert isinstance(self.optimizer, torch.optim.Optimizer)
            checkpoint_dict['optim'] = self.optimizer.state_dict()

            if self.lr_scheduler != None:
                checkpoint_dict['lr_scheduler'] = self.lr_scheduler.state_dict()

            if not os.path.exists(self.config.TRAIN.CHECKPOINT):
                os.mkdir(os.path.join(self.config.TRAIN.CHECKPOINT))

            checkpoint_path = os.path.join(
                self.config.TRAIN.CHECKPOINT, f'checkpoint_{epoch}.pth')
            torch.save(checkpoint_dict, checkpoint_path)

            print(f"saved checkpoint of epoch {epoch}")

    def train_one_epoch(self, epoch):
        """write code of training model at one epoch

        Args:
            step (int): used to summaryWriter, you can use it to mark step in summaryWriter
            epoch (int): epoch of training

        Raises:
            NotImplementedError: if you are not implementing it, you would get exception

        Returns:
            int: return step of this epoch
        Example:
        >>> with tqdm(total=len(self.dataset), leave=True, desc=f"epoch {epoch}",
                ncols=100, unit='it', unit_scale=True) as tbar:
            # train code
            for i, (data, label) in enumerate(self.dataloader):
                data, label = data.cuda(), label.cuda()
                output = model(data) # model is what your want to training
                # calculate loss
                tot_loss = criterion(output, label) 
                # optimizer
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()
                tbar.update()
                tbar.set_postfix(loss=tot_loss, lr=self.optimizer.param_groups[0]['lr'])
            # if you init lr_scheduler, you should write it
            self.lr_scheduler.step(val)
            step += 1 # must write it
        """
        raise NotImplementedError()

    def val_one_epoch(self, epoch):
        raise NotImplementedError()

    def init_criterion(self):
        raise NotImplementedError()

    def init_optimizer(self):
        raise NotImplementedError()
        # config_optim = self.configure_model_params_in_optim(parameters)
        # optimizer = None
        # if optim_str == 'SGD':
        #     optimizer = optim.SGD(parameters, self.config.TRAIN.LR,
        #                           self.config.TRAIN.MOMENTUM, self.config.TRAIN.WD)
        # elif optim_str == 'Adam':
        #     optimizer = optim.Adam(parameters, self.config.TRAIN.LR,
        #                            eps=self.config.TRAIN.EPS, weight_decay=self.config.TRAIN.WD)
        # else:
        #     optimizer = optim.AdamW(
        #         parameters, self.config.TRAIN.LR, weights_decay=self.config.TRAIN.WD)
        # return optimizer

    def init_acc(self):
        raise NotImplementedError()


class TestBaseModel(object):
    def __init__(self, args: argparse.Namespace, config: easydict,
                 dataset: Dataset, models: List[nn.Module], device=torch.device('cpu'), p=0.6):
        super().__init__()
        self.args = args
        self.config = config
        self.models = models
        self.device = device
        self.test_dataset = dataset

    def test(self):
        self.init_dataloader()
        self.init_acc()

        # prepare models
        self.load_pretrained_models()
        if torch.cuda.is_available() and self.config.CUDNN.ENABLED:
            self.models = [m.to(self.device) for m in self.models]

            if self.config.DDP and torch.cuda.device_count() >= 1:
                self.models = [DDP(m, device_ids=[self.args.local_rank],
                                   output_device=self.args.local_rank) for m in self.models]

        # begin testing
        self.test_one_epoch()

    def load_pretrained_models(self):
        if self.config.MODEL.PRETRAINED:
            pretrained_model = torch.load(
                self.config.MODEL.BEST_MODEL, map_location='cpu')
            for i, (model) in enumerate(self.models):
                model.load_state_dict(pretrained_model[f'model_{i}'])

    def test_one_epoch(self):
        raise NotImplementedError()

    def init_acc(self):
        raise NotImplementedError()

    def init_dataloader(self):
        self.test_dataloader = self._init_dataloader(self.test_dataset)

    def _init_dataloader(self, dataset: Dataset) -> DataLoader:
        if self.config.DDP:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset=self.test_dataset,
                                batch_size=self.config.TRAIN.BATCH_SIZE, sampler=sampler, shuffle=False,
                                num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
        return dataloader
