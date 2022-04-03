import torch
from model.simple_baseline import get_post_net_without_res
from config.upt_vcoco_config import config
if __name__ == '__main__':
    pose_net = get_post_net_without_res(config, None)
    a = torch.randn((2, 2048, 55, 47))
    out = pose_net(a)
    print(out.size())