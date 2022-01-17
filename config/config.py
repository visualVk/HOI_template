from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = ''
config.PROJECT_NAME = 'HOI_Template'
config.LOG_DIR = './logs'
config.DATA_DIR = ''
config.GPUS = 1
config.WORKERS = 1
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = 'E:/dataset/'
config.DATASET.DATASET = 'mpii'
config.DATASET.MPII_ORG_ANNO = 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
config.DATASET.MPII_ANNO = 'mpii_human_pose.json'
config.DATASET.TRAIN_ANNO = 'train_joints.csv'
config.DATASET.VAL_ANNO = 'train_joints.csv'
config.DATASET.TEST_ANNO = 'test_joints.csv'
config.DATASET.TRAIN_SET = 'images'
config.DATASET.TEST_SET = 'test_joints.csv'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'SGD'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 5

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = './data/'
config.TRAIN.SAVE_BEGIN = 40
config.TRAIN.INTERVAL_SAVE = 5

config.TRAIN.BATCH_SIZE = 1
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config_by_yaml(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
        config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
        config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
        config.DATA_DIR, config.MODEL.PRETRAINED)
