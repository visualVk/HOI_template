from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LPN(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 hidden_num: int,
                 num_joints: int,
                 heatmap_size: Union[int,
                                     Tuple[int,
                                           int]]):
        super(LPN, self).__init__()
        self.joint_lc = self._make_ly_of_joints(
            num_joints, hidden_size, heatmap_size)

    def _make_ly_of_joints(self, num_joints, hidden_size,
                           heatmap_size: Union[int, Tuple[int, int]]):
        representation_size = num_joints * (heatmap_size * heatmap_size if isinstance(
            heatmap_size, int) else heatmap_size[0] * heatmap_size[1])
        layers = []
        layers.append(nn.Linear(representation_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, heatmap: torch.Tensor, obj_lc_feat: torch.Tensor):
        heatmap_flattened = heatmap.view(-1,
                                         heatmap.size(1) * heatmap.size(2) * heatmap.size(3))
        human_lc_feat = self.joint_lc(heatmap_flattened)
        # print(human_lc_feat.shape, obj_lc_feat.shape)
        hs_feat = torch.cat([human_lc_feat, obj_lc_feat],
                            dim=0).to(heatmap_flattened.device)
        return hs_feat


def build_lpn_model(hidden_size, hidden_num, num_joints, heatmap_size):
    lpn = LPN(hidden_size, hidden_num, num_joints, heatmap_size)
    return lpn


class LPN_2(nn.Module):
    def __init__(self) -> None:
        super(LPN_2, self).__init__()
        self.conv1 = nn.Conv2d(256, 2048, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, 8, 8)
        return x
