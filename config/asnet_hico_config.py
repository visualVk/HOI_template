from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import yaml
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_ROOT = 'data/'
config.PROJECT_NAME = "asnet_hico"
config.LOG_DIR = './logs'
config.DATA_DIR = ''
config.WORKERS = 1
config.PRINT_FREQ = 20
config.SEED = 42
config.AUX_LOSS = False
config.EVAL = False

config.MODEL = edict()
# specific model
config.MODEL.FILE = ''
config.MODEL.NAME = ''
# resume
config.MODEL.RESUME_PATH = 'data/checkpoint/checkpoint_0.pth'
config.MODEL.MASKS = False

# backbone
config.BACKBONE = edict()
config.BACKBONE.NAME = 'resnet50'
config.BACKBONE.DIALATION = False

# transformer
config.TRANSFORMER = edict()
config.TRANSFORMER.BRANCH_AGGREGATION = False
config.TRANSFORMER.POSITION_EMBEDDING = 'sine' # choices=('sine', 'learned')
config.TRANSFORMER.HIDDEN_DIM = 256
config.TRANSFORMER.ENC_LAYERS = 6
config.TRANSFORMER.DEC_LAYERS = 6
config.TRANSFORMER.DIM_FEEDFORWARD = 2048
config.TRANSFORMER.DROPOUT = 0.1
config.TRANSFORMER.NHEADS = 8
config.TRANSFORMER.NUM_QUERIES = 100
config.TRANSFORMER.REL_NUM_QUERIES = 16
config.TRANSFORMER.PRE_NORM = False

# matcher
config.MATCHER = edict()
config.MATCHER.COST_CLASS = 1
config.MATCHER.COST_BBOX = 5
config.MATCHER.COST_GIOU = 2

# LOSS
config.LOSS = edict()
config.LOSS.AUX_LOSS = True
config.LOSS.DICE_LOSS_COEF = 1
config.LOSS.DET_CLS_COEF = [1, 1]
config.LOSS.REL_CLS_COEF = 1
config.LOSS.BBOX_LOSS_COEF = [5, 5]
config.LOSS.GIOU_LOSS_COEF = [2, 2]
config.LOSS.EOS_COEF = 0.1
# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = False
config.CUDNN.DETERMINISTIC = True
config.CUDNN.ENABLED = True

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = 'data/hico/hico/images'
config.DATASET.NAME = 'HICODetDataset'
config.DATASET.MEAN = [0.485, 0.456, 0.406]
config.DATASET.STD = [0.229, 0.224, 0.225]
config.DATASET.MAX_SIZE = 1333
config.DATASET.SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
config.DATASET.IMG_NUM_PER_GPU = 2
config.DATASET.SUB_NUM_CLASSES = 1
config.DATASET.OBJ_NUM_CLASSES = 91
config.DATASET.REL_NUM_CLASSES = 117

# Matcher
config.MATCHER = edict()
config.MATCHER.COST_CLASS = 1
config.MATCHER.COST_BBOX = 5
config.MATCHER.COST_GIOU = 2

# LOSS
config.LOSS = edict()
config.LOSS.AUX_LOSS = True
config.LOSS.DICE_LOSS_COEF = 1
config.LOSS.DET_CLS_COEF = [1, 1]
config.LOSS.REL_CLS_COEF = 1
config.LOSS.BBOX_LOSS_COEF = [5, 5]
config.LOSS.GIOU_LOSS_COEF = [2, 2]
config.LOSS.EOS_COEF = 0.1

# trainer
config.TRAINER = edict()
config.TRAINER.FILE = ''
config.TRAINER.NAME = ''

# train
config.TRAIN = edict()
config.TRAIN.OPTIMIZER = ''
config.TRAIN.LR = 0.0001
config.TRAIN.LR_BACKBONE = 0.00001
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WEIGHT_DECAY = 0.0001
# optimizer SGD
config.TRAIN.NESTEROV = False
# learning rate scheduler
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_DROP = 55
config.TRAIN.CLIP_MAX_NORM = 0.1
# config.TRAIN.MAX_EPOCH = 100
# train resume
config.TRAIN.RESUME = False
config.TRAIN.BEGIN_EPOCH = 1
config.TRAIN.END_EPOCH = 100
config.TRAIN.CHECKPOINT = "data/checkpoint"
# print freq
config.TRAIN.PRINT_FREQ = 20
# save checkpoint during train
config.TRAIN.SAVE_INTERVAL = 5
config.TRAIN.SAVE_EVERY_CHECKPOINT = True
# val when train
config.TRAIN.VAL_WHEN_TRAIN = False

# test
config.TEST = edict()
config.TEST.REL_ARRAY_PATH = 'data/hico/hico/rel_np.npy'
config.TEST.USE_EMB = True
config.TEST.MODE = 'hico'


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


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
