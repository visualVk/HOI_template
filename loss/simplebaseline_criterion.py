from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # print(output.shape, target.shape)
        heatmaps_pred = output.view(batch_size, num_joints, -1).split(1, 1)
        heatmaps_gt = target.view(batch_size, num_joints, -1).split(1, 1)
        # print(heatmaps_pred[0].shape, heatmaps_gt[0].shape)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                # print(heatmap_pred.shape, target_weight[:, idx].shape)
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsL1Loss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsL1Loss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        pred_joints = output.view(batch_size, num_joints, -1).split(1, 1)
        gt_joints = target.view(batch_size, num_joints, -1).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            pred_joint = pred_joints[idx].squeeze()
            gt_joint = gt_joints[idx].squeeze()
            if self.use_target_weight:
                # print(heatmap_pred.shape, target_weight[:, idx].shape)
                loss += 0.5 * self.criterion(
                    pred_joint.mul(target_weight[:, idx]),
                    gt_joint.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(pred_joint, gt_joint)

        return loss / num_joints
