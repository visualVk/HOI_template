import argparse

from model.SMPN import SMPN
from dataset.SimpleDataset import RandomDataset
from model.SimpleNet import Model
import easydict


def prepare_train(args: argparse.Namespace, config: easydict):
    model = Model(5, 2)
    dataset = RandomDataset(5, 10)
    train_base_model = SMPN(args, config, dataset, [model])
    return train_base_model, model, dataset


def train(args: argparse.Namespace, config: easydict):
    train_base_model, model, dataset = prepare_train(args, config)
    train_base_model.train()
