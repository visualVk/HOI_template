from typing import Union, Tuple

import torch
from torch import nn


class LPN(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 human_hidden_size: int,
                 hidden_num: int,
                 num_joints: int,
                 heatmap_size: Union[int,
                                     Tuple[int,
                                           int]]):
        super(LPN, self).__init__()
        assert hidden_size >= human_hidden_size, "wrong hidden_size of human_hidden_size, hidden_size must greater than human_hidden_size"
        obj_hidden_size = hidden_size - human_hidden_size
        self.obj_hidden_size = obj_hidden_size
        self.hidden_size = hidden_size
        self.num_joints = num_joints
        self.hidden_num = hidden_num
        self.joint_lc = self._make_ly_of_joints(
            num_joints, hidden_num, heatmap_size)

    def _make_ly_of_joints(self, num_joints, hidden_num,
                           heatmap_size: Union[int, Tuple[int, int]]):
        representation_size = num_joints * (heatmap_size * heatmap_size if isinstance(
            heatmap_size, int) else heatmap_size[0] * heatmap_size[1])
        layers = []
        layers.append(nn.Linear(representation_size, hidden_num))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_num))
        return nn.Sequential(*layers)

    def _make_concat_zeros(self, hm_lc_feat: torch.Tensor):
        obj_lc_feat = torch.zeros(
            (self.obj_hidden_size,
             self.hidden_num),
            device=hm_lc_feat.device)
        hs_feat = torch.cat([hm_lc_feat, obj_lc_feat]).to(hm_lc_feat.device)
        return hs_feat

    def forward(self, heatmap: torch.Tensor):
        heatmap_flattened = heatmap.view(-1,
                                         heatmap.size(1) * heatmap.size(2) * heatmap.size(3))
        human_lc_feat = self.joint_lc(heatmap_flattened)
        hs_feat = self._make_concat_zeros(human_lc_feat)
        return hs_feat
