import argparse
from argparse import Namespace

import easydict
import torch
import torch.nn as nn
from torch import Tensor

from utils.model import adapt_device


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.avg_pool = nn.Conv2d(3, 2, 1)

    def forward(self, input: Tensor):
        x = self.avg_pool(input)
        return x


def build_simplenet(
        config: easydict.EasyDict,
        args: argparse.Namespace):
    cuda = config.CUDNN.ENABLED
    ddp = config.DDP
    local_rank = args.local_rank

    net = Model()
    cuda = cuda and torch.cuda.is_available()
    ddp = ddp and cuda
    net = adapt_device(net, ddp, cuda, local_rank)
    return net
