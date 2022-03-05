import argparse
from typing import Union, Tuple, List, Dict, Any

import easydict
import torch
from torch import nn, Tensor
from torchvision.ops import batched_nms

from utils import box_ops
from utils.model import adapt_device
from . import NestedTensor
from .detr import DETR, build_detr
from .ds import nested_tensor_from_tensor_list
from .gcn import GCN


class InteractionHead(nn.Module):
    def __init__(
            self,
            detr: DETR,
            postprocessor,
            human_id=0,
            hidden_dim=256,
            num_classes=80,
            verb_classes=117,
            interaction_class=600,
            box_score_thresh=0.2,
            fg_iou_thresh=0.5,
            min_instances=3,
            max_instances=15):
        super(InteractionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances

        self.detr = detr
        self.postprocessor = postprocessor
        self.gcn = GCN(human_id, hidden_dim, representation_size=hidden_dim)
        self.human_id = human_id
        self.verb_embed = nn.Linear(hidden_dim, verb_classes)

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, self.hidden_dim)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_id
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum()
            n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified
            # interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
            ))

        return region_props

    def forward(self, images: Tensor):
        bs = images.size(0)
        image_shapes = torch.tensor(
            [im.shape[-2:] for im in images], device=images.device).view(bs, 2)

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(
                [image for image in images])

        # >>>>>>>>>>>>>>>>>>>> transformer <<<<<<<<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(images)

        src, mask = features[-1].decompose()
        assert mask is not None
        input_proj = self.detr.input_proj(src)
        # hs: [enc_layers, bs, c, hidden_states]
        hs = self.detr.transformer(input_proj, mask,
                                   self.detr.query_embed.weight, pos[-1])[0]
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        # >>>>>>>>>>>>>>>>>>>>            <<<<<<<<<<<<<<<<<<<<<<

        boxes, scores, labels, hidden_states = [], [], [], []
        for output_class, output_coord, hds, image_shape in zip(
                outputs_class, outputs_coord, hs[-1], image_shapes):
            hds = hds.unsqueeze(dim=0)
            results = {
                'pred_logits': output_class[-1].unsqueeze(dim=0),
                'pred_boxes': output_coord[-1].unsqueeze(dim=0)
            }
            results = self.postprocessor(results, image_shape.unsqueeze(dim=0))
            region_props = self.prepare_region_proposals(results, hds)
            box, score, label, h = region_props[0].values()
            boxes.append(box)
            scores.append(score)
            labels.append(label)
            hidden_states.append(h)

        # >>>>>>>>>>>>>>>>>>>> starting predict action between humans and objec
        # hs: [bs, enc_layers * c, hidden_states]
        verb_list, human_list, object_list, human_coord_list, object_coord_list = [], [], [], [], []
        for label, n_node, coord, image_shape in zip(
                labels, hidden_states, boxes, image_shapes):
            shape = image_shape
            if isinstance(image_shape, Tensor):
                shape = (image_shape[0], image_shape[1])
            n_h = torch.sum(self.human_id == label)
            n = label.size(0)

            if not torch.all(label[:n_h] == self.human_id):
                h_id = torch.nonzero(label == self.human_id).squeeze(1)
                o_id = torch.nonzero(label != self.human_id).squeeze(1)
                perm = torch.cat([h_id, o_id])
                label = label[perm]
                coord = coord[perm]
                n_node = n_node[perm]

            if n_h < 1 or n == 1:
                verb_list.append(torch.zeros(0, dtype=torch.int64))
                human_list.append(torch.zeros(0, dtype=torch.int64))
                object_list.append(torch.zeros(0, dtype=torch.int64))
                human_coord_list.append(torch.zeros(0, dtype=torch.int64))
                object_coord_list.append(torch.zeros(0, dtype=torch.int64))
                continue

        # >>>>>>>>>>>>>>>>>>>>> GCN Interaction Head <<<<<<<<<<<<<<<<<<<<<<
            node, human_node, unary, x_keep, y_keep = self.gcn(
                label, n_node, coord, shape)

            unary = unary.unsqueeze(dim=1)
            verb_feature = node * unary * human_node
            verb_feature = verb_feature.to(images.device)
            verb_label = self.verb_embed(verb_feature)

            box_h = coord[:n_h][x_keep]
            box_o = coord[y_keep]
            box_h = self.recover_boxes(box_h, image_shape)
            box_o = self.recover_boxes(box_o, image_shape)

            verb_list.append(verb_label)
            human_list.append(label[:n_h][x_keep])
            object_list.append(label[y_keep])
            human_coord_list.append(box_h)
            object_coord_list.append(box_o)

        result = dict(
            verbs=verb_list,
            boxes_h=human_coord_list,
            boxes_o=object_coord_list,
            labels_h=human_list,
            labels_o=object_list
        )

        return result


def build_interaction_net(
        config: easydict.EasyDict,
        args: argparse.Namespace):
    cuda = config.CUDNN.ENABLED
    ddp = config.DDP
    local_rank = args.local_rank

    cuda = cuda and torch.cuda.is_available()
    ddp = ddp and cuda

    detr, criterion, postprocessors = build_detr(config, args)
    model_without_ddp = InteractionHead(
        detr, postprocessors['bbox'], human_id=1)
    model_without_ddp, model = adapt_device(
        model_without_ddp, ddp, cuda, local_rank)
    return model_without_ddp, model
