import torch.nn as nn
import argparse
import os
import torch
import easydict
from typing import List
import utils.misc as utils
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


class BaseModel(object):
    def __init__(self, args: argparse.Namespace, config: easydict,
                 dataset: Dataset, models: List[nn.Module]):
        super().__init__()
        self.args = args
        self.config = config
        self.models = models
        self.dataset = dataset

    def train(self):
        # prepare dataset and dataloader
        sampler = DistributedSampler(self.dataset)
        self.dataloader = DataLoader(
            dataset=self.dataset, batch_size=self.config.TRAIN.BATCH_SIZE,
            sampler=sampler, shuffle=False)

        # prepare models
        if torch.cuda.is_available() and self.config.CUDNN.ENABLED:
            self.models = [m.cuda() for m in self.models]

            if torch.cuda.device_count() >= 1:
                self.models = [DDP(m, device_ids=[self.args.local_rank],
                                   output_device=self.args.local_rank) for m in self.models]

        self.init_optimizer()
        self.init_criterion()

        # begin training
        begin_epoch = self.config.TRAIN.BEGIN_EPOCH
        end_epoch = self.config.TRAIN.END_EPOCH

        step = 0
        for epoch in range(begin_epoch, end_epoch):
            self.dataloader.sampler.set_epoch(epoch)
            step = self.train_one_epoch(step, epoch)

            # save checkpoint
            if utils.is_main_process() \
                    and epoch >= self.config.TRAIN.SAVE_BEGIN \
                    and epoch % self.config.TRAIN.INTERVAL_SAVE == 0:
                model_save = {}
                optim_save = {}

                for i in range(len(self.models)):
                    model = self.models[i].module
                    model_save[f'model_{i}'] = model.state_dict()

                optim_save['optim'] = self.optimizer.state_dict()

                save_dict = {**model_save, **optim_save}

                if not os.path.exists(self.config.TRAIN.CHECKPOINT):
                    os.mkdir(os.path.join(self.config.TRAIN.CHECKPOINT))

                checkpoint_path = os.path.join(
                    self.config.TRAIN.CHECKPOINT, f'checkpoint_{self.args.local_rank}_{epoch}.pth')
                torch.save(save_dict, checkpoint_path)

    def train_one_epoch(self, step, epoch) -> int:
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

    def reload_models(self):
        if self.config.TRAIN.RESUME:
            checkpoint = torch.load(os.path.join(self.config.TRAIN.CHECKPOINT,
                                                 f"checkpoint_{self.config.TRAIN.BEGIN_EPOCH}.pth"))
            for i, (m) in enumerate(self.models):
                m.load_state_dict(checkpoint[f"model_{i}"])
            self.optimizer.load_state_dict(checkpoint['optim'])

    def init_criterion(self):
        raise NotImplementedError()

    def init_optimizer(self):
        raise NotImplementedError()

    def init_scheduler(self):
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
