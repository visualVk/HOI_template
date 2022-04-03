import argparse
from typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP
from model.simple_net import build_simplenet
from model.backbone import *
from model.position_encoding import *
from model.detr.detr import build_detr

ModelFactory = dict(
    simple_net=build_simplenet,
    detr=build_detr,
)


def build_model(model_name: str,
                config: easydict.EasyDict,
                args: argparse.Namespace) -> Union[nn.Module,
                                                   nn.DataParallel,
                                                   DDP]:
    assert model_name in ModelFactory.keys(),\
        f"{model_name} is not in f{ModelFactory.keys()}"
    create_fn = ModelFactory[model_name]
    return create_fn(config, args)
