import argparse
import gc
from model.SMPN import TrainSMPN
from dataset.SimpleDataset import RandomDataset
from model.SimpleNet import Model
import easydict


def prepare_train(args: argparse.Namespace, config: easydict):
    model = Model(5, 2)
    train_dataset = RandomDataset(5, 10) # below 100, label doesn't have negative number
    val_dataset = RandomDataset(5, 10)
    train_base_model = TrainSMPN(args, config, train_dataset, [model])
    # train_base_model = TrainSMPN.init_with_train_and_val(
        # args, config, train_dataset, val_dataset, [model])
    return train_base_model, model, train_dataset


def train(args: argparse.Namespace, config: easydict):
    train_base_model, model, dataset = prepare_train(args, config)
    train_base_model.train()

    del train_base_model
    gc.collect()
