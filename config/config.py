from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import yaml
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = ''
config.PROJECT_NAME = 'HOI_Template'
config.LOG_DIR = './logs'
config.DATA_DIR = ''
config.GPUS = 1
config.WORKERS = 1
config.PRINT_FREQ = 20
config.SEED = 2112112047
config.AUX_LOSS = False
config.DDP = True  # debug must be False

# Model
config.MODEL = edict()
config.MODEL.BACKBONE = 'resnet50'
config.MODEL.NAME = "DETR"
config.MODEL.HIDDEN_DIM = 256
config.MODEL.POSITION_EMB = 'sine'
config.MODEL.NHEAD = 8
config.MODEL.NUM_QUERIES = 100
config.MODEL.ENC_LAYERS = 6
config.MODEL.DEC_LAYERS = 6
config.MODEL.DROPOUT = 0.1
config.MODEL.AUX_LOSS = False
config.MODEL.PRE_NORM = False
config.MODEL.DIM_FEEDFORWARD = 2048
config.MODEL.BEST_MODEL = './data'
config.MODEL.PRETRAINED = False

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = False
config.CUDNN.DETERMINISTIC = True
config.CUDNN.ENABLED = True

# DATASET related params
config.DATASET = edict()
# config.DATASET.ROOT = '/root/autodl-tmp/data/'
config.DATASET.ROOT = './data'
config.DATASET.NAME = 'hico'
config.DATASET.IMAGES_TRAIN = os.path.join(config.DATASET.ROOT, config.DATASET.NAME, 'images/train2015/')
config.DATASET.IMAGES_TEST = os.path.join(config.DATASET.ROOT, config.DATASET.NAME, 'images/test2015/')
config.DATASET.ANNO = 'anno/hico_trainval_remake.odgt'
config.DATASET.ANNO_TRAIN = os.path.join(config.DATASET.ROOT, config.DATASET.NAME, 'anno/hico_train.json')
config.DATASET.ANNO_TEST = os.path.join(config.DATASET.ROOT, config.DATASET.NAME, 'anno/hico_test.json')
config.DATASET.INTERACTION_NAME = os.path.join(config.DATASET.ROOT, config.DATASET.NAME, 'anno/hico_verb_names.json')

# Matcher
config.MATCHER = edict()
config.MATCHER.COST_CLASS = 1
config.MATCHER.COST_BBOX = 5
config.MATCHER.COST_GIOU = 2

# Criterion
config.CRITERION = edict()
config.CRITERION.DICE_LOSS_COEF = 1
config.CRITERION.BBOX_LOSS_COEF = 5
config.CRITERION.GIOU_LOSS_COEF = 2
config.CRITERION.EOS_COEF = 0.02

# Train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = 200
config.TRAIN.LR = 1e-4
config.TRAIN.LR_BACKBONE = 1e-5

config.TRAIN.OPTIMIZER = 'SGD'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 1e-4
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.CLIP_MAX_NORM = 0.1

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 10

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = './data/checkpoint'
config.TRAIN.SAVE_BEGIN = 4
config.TRAIN.INTERVAL_SAVE = 1

config.TRAIN.BATCH_SIZE = 2
config.TRAIN.SHUFFLE = False

# Test
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
