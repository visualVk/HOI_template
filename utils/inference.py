import os
import torch
from utils import visual
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff

from mpl_toolkits.axes_grid1 import make_axes_locatable

# from utils import DataFactory
# from upt import build_detector

warnings.filterwarnings("ignore")


def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(
            a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i + 1), fontsize=20,
                       fontweight='semibold', color='w')
        txt.set_path_effects(
            [peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()


def visualise_entire_image(image, output, actions, action=None, thresh=0.2):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique())
    no = len(boxes)

    scores = output['scores']
    pred = output['labels']
    # Visualise detected human-object pairs with attached scores
    if action is not None:
        keep = torch.nonzero(
            torch.logical_and(
                scores >= thresh,
                pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        visual.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')

        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i],
                                 :2],
                           f"{scores[keep[i]]:.2f}",
                           fontsize=15,
                           fontweight='semibold',
                           color='w')
            txt.set_path_effects(
                [peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        plt.show()
        return

    pairing = output['pairing']
    coop_attn = output['attn_maps'][0]
    comp_attn = output['attn_maps'][1]

    # Visualise attention from the cooperative layer
    for i, attn_1 in enumerate(coop_attn):
        fig, axe = plt.subplots(2, 4)
        fig.suptitle(f"Attention in coop. layer {i}")
        axe = np.concatenate(axe)
        ticks = list(range(attn_1[0].shape[0]))
        labels = [v + 1 for v in ticks]
        for ax, attn in zip(axe, attn_1):
            im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)

    x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    x, y = torch.nonzero(x != y).unbind(1)
    pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]

    # Visualise attention from the competitive layer
    fig, axe = plt.subplots(2, 4)
    fig.suptitle("Attention in comp. layer")
    axe = np.concatenate(axe)
    ticks = list(range(len(pairs)))
    for ax, attn in zip(axe, comp_attn):
        im = ax.imshow(attn, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(pairs, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(pairs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    # Print predicted actions and corresponding scores
    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {actions[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx] + 1
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                f"score: {scores[idx]:.4f}"
            )

    # Draw the bounding boxes
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    draw_boxes(ax, boxes)
    plt.show()
