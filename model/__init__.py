import argparse
from typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP
from model.backbone import *
from model.position_encoding import *

ModelFactory = dict(
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
