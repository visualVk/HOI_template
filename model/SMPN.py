import argparse
import easydict
import utils.misc as utils
from typing import List
from torch import nn, optim
import torch
from tqdm import tqdm
from model.BaseModel import TrainBaseModel, TestBaseModel
from torch.utils.data import Dataset, DataLoader


class TrainSMPN(TrainBaseModel):
    def __init__(self, args: argparse.Namespace, config: easydict, dataset, models: List[nn.Module], p=0.6):
        super(TrainSMPN, self).__init__(
            args, config, dataset, models, p)

    # @classmethod
    # def init_with_train_and_val(cls, args: argparse.Namespace, config: easydict,
    #                             train_dataset: Dataset, val_dataset: Dataset, models: List[nn.Module]):
    #     obj = super(TrainSMPN, cls).init_with_train_and_val(
    #         args, config, train_dataset, val_dataset, models)
    #     print(type(obj))
    #     return obj

    def init_criterion(self):
        l1 = nn.CrossEntropyLoss()
        self.criterion = [l1]

    def init_optimizer(self):
        model = self.models[0].module
        self.optimizer = optim.SGD(
            [{"params": model.parameters(), "lr": self.config.TRAIN.LR}])

    def _cal_loss(self, x, y):
        return self.criterion[0](x, y)

    def train_one_epoch(self, step, epoch):
        with tqdm(total=len(self.train_dataloader), leave=True, desc=f"train epoch {epoch}",
                  ncols=100, unit='it', unit_scale=True) as tbar:

            model_ddp = self.models[0]
            model_ddp.train()
            for i, (data, label) in enumerate(self.train_dataloader):
                if torch.cuda.is_available() and self.config.CUDNN.ENABLED:
                    data, label = data.cuda(), label.long().cuda()
                output = model_ddp(data)
                tot_loss = self._cal_loss(output, label)

                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()

                tbar.update()
                tbar.set_postfix(
                    loss=tot_loss.detach().cpu().item(), lr=self.optimizer.param_groups[0]['lr'])
                step += 1

        return step

    def val_one_epoch(self, step, epoch):
        with tqdm(total=len(self.val_dataloader), leave=True, desc=f"eval epoch {epoch}",
                  ncols=100, unit='it', unit_scale=True) as tbar:

            model_ddp = self.models[0]
            model_ddp.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(self.val_dataloader):
                    if torch.cuda.is_available() and self.config.CUDNN.ENABLED:
                        data, label = data.cuda(), label.long().cuda()
                    output = model_ddp(data)
                    tot_loss = self._cal_loss(output, label)

                    tbar.update()
                    tbar.set_postfix(
                        loss=tot_loss.detach().cpu().item())
                    step += 1

        return step


class TestSMPN(TestBaseModel):

    def __init__(self, args: argparse.Namespace, config: easydict, dataset, models: List[nn.Module]):
        super(TestSMPN, self).__init__(args, config, dataset, models)

    def init_acc(self):
        self.criterion = [nn.CrossEntropyLoss()]

    def _cal_loss(self, x, y):
        return self.criterion[0](x, y)

    def test_one_epoch(self, step):
        with tqdm(total=len(self.test_dataloader), leave=True, desc=f"test",
                  ncols=100, unit='it', unit_scale=True) as tbar:

            model_ddp = self.models[0]
            model_ddp.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(self.test_dataloader):
                    if torch.cuda.is_available() and self.config.CUDNN.ENABLED:
                        data, label = data.cuda(), label.long().cuda()
                    output = model_ddp(data)
                    tot_loss = self._cal_loss(output, label)

                    tbar.update()
                    tbar.set_postfix(
                        loss=tot_loss.detach().cpu().item())
                    step += 1