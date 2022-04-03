from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def calc_dists(preds, target, normalize):
    normalize = normalize.to(target.device)
    preds = preds.to(torch.float32).to(target.device)
    target = target.to(torch.float32)
    dists = torch.zeros((preds.shape[1], preds.shape[0]))
    # print(preds.shape, target.shape)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = torch.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = torch.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    target should be in 64x64
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        # target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = torch.ones((pred.shape[0], 2)) * torch.tensor([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = torch.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def get_max_preds(batch_heatmaps: torch.Tensor):

    assert isinstance(batch_heatmaps, torch.Tensor), \
        'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.size()) == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.view(batch_size, num_joints, -1)
    # index = row * width + col
    idx = torch.argmax(heatmaps_reshaped, 2)  # n x 16 x 1 ==> max point index
    # n x 16 x 1 ==> max point value
    maxvals = torch.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.view(batch_size, num_joints, 1)
    idx = idx.view(batch_size, num_joints, 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2)
    pred_mask = pred_mask.float()

    preds *= pred_mask
    return preds, maxvals


def generate_target(joints, image_size):
    '''
    :param joints:  [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    # TODO: add values in config
    num_joints = 17
    heatmap_size = torch.tensor([64, 64])
    sigma = 2

    target_weight = torch.ones((num_joints, 1), dtype=torch.float32)
    # print(joints[:, -1].shape)
    target_weight[:, 0] = joints[:, -1]

    # if target_type == 'gaussian':
    target = torch.zeros(
        (num_joints,
         heatmap_size[0],
         heatmap_size[1]),
        dtype=torch.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = torch.tensor(image_size) / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            # target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(0, size, 1, dtype=torch.float32)
        y = x[:, None]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal
        # 1
        g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                      (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight
