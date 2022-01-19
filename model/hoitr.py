import argparse
from typing import List
import torch.nn as nn
import easydict
import torch.nn.functional as F
import torch
from torch.nn import Transformer
from loss.hoi_criterion import SetCriterion
from model.transformer import build_transformer
from model.backbone import build_backbone
from utils.hoi_matcher import build_matcher as build_hoi_matcher
from model.nested_tensor import NestedTensor, nested_tensor_from_tensor_list

num_humans = 2


class HoiTR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone: nn.Module, transformer: Transformer,
                 num_classes: int, num_actions: int, num_queries: int, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.human_cls_embed = nn.Linear(hidden_dim, num_humans + 1)
        self.human_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.object_cls_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.object_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.action_cls_embed = nn.Linear(hidden_dim, num_actions + 1)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # features: [feat + mask], only fetch the highest level feature
        assert isinstance(features, List[NestedTensor])
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask,
                              self.query_embed.weight, pos[-1])[0]

        human_outputs_class = self.human_cls_embed(hs)
        human_outputs_coord = self.human_box_embed(hs).sigmoid()
        object_outputs_class = self.object_cls_embed(hs)
        object_outputs_coord = self.object_box_embed(hs).sigmoid()
        action_outputs_class = self.action_cls_embed(hs)

        out = {
            'human_pred_logits': human_outputs_class[-1],
            'human_pred_boxes': human_outputs_coord[-1],
            'object_pred_logits': object_outputs_class[-1],
            'object_pred_boxes': object_outputs_coord[-1],
            'action_pred_logits': action_outputs_class[-1],
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                human_outputs_class,
                human_outputs_coord,
                object_outputs_class,
                object_outputs_coord,
                action_outputs_class,
            )
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args: argparse.Namespace, config: easydict.EasyDict):
    assert config.DATASET.NAME in [
        'hico', 'vcoco', 'hoia'], config.DATASET.NAME
    if config.DATASET.NAME in ['hico']:
        num_classes = 91
        num_actions = 118
    elif config.DATASET.NAME in ['vcoco']:
        num_classes = 91
        num_actions = 30
    else:
        num_classes = 12
        num_actions = 11

    # device = torch.device(args.device)

    # if config.MODEL.BACKBONE == 'swin':
    #     from .backbone_swin import build_backbone_swin
    #     backbone = build_backbone_swin(args)
    # else:
    backbone = build_backbone(config)

    transformer = build_transformer(config)

    model = HoiTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_actions=num_actions,
        num_queries=config.MODEL.NUM_QUERIES,
        aux_loss=config.MODEL.AUX_LOSS,
    )

    return model
