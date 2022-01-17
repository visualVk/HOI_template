import argparse
import gc
from model.SMPN import TestSMPN
from dataset.SimpleDataset import RandomDataset
from model.SimpleNet import Model
import easydict


def prepare_test(args: argparse.Namespace, config: easydict):
    model = Model(5, 2)
    dataset = RandomDataset(5, 10)
    test_base_model = TestSMPN(args, config, dataset, [model])
    return test_base_model, model, dataset


def test(args: argparse.Namespace, config: easydict):
    test_base_model, model, dataset = prepare_test(args, config)
    test_base_model.test()

    del test_base_model
    gc.collect()
