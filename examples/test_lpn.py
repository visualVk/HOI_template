import torch

from model.lpn import LPN

if __name__ == '__main__':
    lpn = LPN(6, 3, 100, 17, 64)
    joints_feat = torch.randn(3, 17, 64, 64)
    output = lpn(joints_feat)
    print(output.shape)