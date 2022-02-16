from argparse import Namespace
from typing import Dict, Union
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from .simple_net import build_simplenet
from .backbone import *
from .position_encoding import *
from transforms import *

from config import config as Cfg
ModelFactory = dict(
    simple_net=build_simplenet
)


def build_model(model_name: str, cuda: bool, ddp: bool,
                local_rank: int) -> Union[nn.Module, nn.DataParallel, DDP]:
    assert model_name in ModelFactory.keys(),\
        f"{model_name} is not in f{ModelFactory.keys()}"
    create_fn = ModelFactory[model_name]
    return create_fn(cuda, ddp, local_rank)
