import argparse
import easydict
import torch
from model.hoitr import build as build_model
from model.hoi_base_model import HoiTrainModel
from dataset import build_dataset


def train(args: argparse.Namespace, config: easydict.EasyDict):
    model = build_model(args, config)
    dataset = build_dataset(image_set='train', config=config)
    device = torch.device('cpu')
    if torch.cuda.is_available() and config.CUDNN.ENABLED:
        if config.DDP:
            device = torch.device(args.local_rank)
        else:
            device = torch.device('cuda')
    train_base_model = HoiTrainModel(args, config, dataset, [model], device)
    train_base_model.train()
