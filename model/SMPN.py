import argparse
import easydict
import utils.misc as utils
from typing import List
from torch import nn, optim
import torch
from tqdm import tqdm
from model.BaseModel import BaseModel
from torch.utils.data import Dataset, DataLoader


class SMPN(BaseModel):
    def __init__(self, args: argparse.Namespace, config: easydict, dataset, models: List[nn.Module]):
        super(SMPN, self).__init__(args, config, dataset, models)

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
        with tqdm(total=len(self.dataloader), leave=True, desc=f"epoch {epoch}",
                  ncols=100, unit='it', unit_scale=True) as tbar:

            model_ddp = self.models[0]
            for i, (data, label) in enumerate(self.dataloader):
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
