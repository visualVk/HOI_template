import torch
from model.simple_baseline import get_post_net_without_res
from config.upt_vcoco_config import config
if __name__ == '__main__':
    cfg = config
    simple_baseline = get_post_net_without_res(config, None)