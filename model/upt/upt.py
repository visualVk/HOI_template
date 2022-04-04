from model.ds.nested_tensor import nested_tensor_from_tensor_list
from utils import box_ops
from model.detr.detr import build_detr as build_model
from model.simple_baseline import get_post_net_without_res
from loss.simplebaseline_criterion import JointsMSELoss
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from utils.keypoint import accuracy, generate_target
from typing import Optional, List, Tuple
from torchvision.ops.boxes import batched_nms, box_iou

from utils.ops import binary_focal_loss_with_logits
from model.upt.interaction_head import InteractionHead

import sys
sys.path.append('detr')


class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """

    def __init__(self,
                 detector: nn.Module,
                 postprocessor: nn.Module,
                 interaction_head: nn.Module,
                 human_idx: int, num_classes: int,
                 pose_net: Optional[nn.Module] = None,
                 alpha: float = 0.5, gamma: float = 2.0,
                 box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
                 min_instances: int = 3, max_instances: int = 15,
                 ) -> None:
        super().__init__()
        self.detector = detector
        self.pose_net = pose_net

        self.postprocessor = postprocessor
        self.interaction_head = interaction_head
        self.criterion = JointsMSELoss()

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)
        # gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        # gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        gt_bx_h = targets['boxes_h']
        gt_bx_o = targets['boxes_o']

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets['labels'][y]] = 1

        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / (n_p + 1 if n_p == 0 else n_p)

    def compute_keypoint_loss(
            self,
            heatmaps,
            targets,
            images_size):
        # list[[17, 64, 64]]
        images_size = images_size.detach().cpu().numpy().tolist()
        # print(targets.shape, heatmaps.shape)
        loss = torch.tensor([0], dtype=torch.float32, device='cuda')
        # n_p = 0
        for i, heatmap in enumerate(heatmaps):
            for j, image_size in enumerate(images_size):
                ratios = tuple(float(64) / float(s_orig)
                               for s_orig in image_size)
                ratio_weight, ratio_height = ratios
                # 17 should be instead of parameter in config
                scaled_targets = targets[j]["keypoints"].view(-1, 17, 3) * torch.as_tensor(
                    [ratio_weight, ratio_height, 1], device=targets[i]["keypoints"].device)
                if scaled_targets.size(0) == 0:
                    continue
                if heatmap.size(0) == 0:
                    heatmap = torch.zeros(
                        (scaled_targets.size(0),
                         scaled_targets.size(1),
                         heatmap.size(2),
                            heatmap.size(3)))
                else:
                    limited_size = min(
                        scaled_targets.shape[0], heatmap.shape[0])
                    heatmap = heatmap[:limited_size]
                    scaled_targets = scaled_targets[:limited_size]
                # print(heatmap.shape, targets[j]["keypoints"].shape)
                if not self.training:
                    scaled_targets = scaled_targets[:, :, :2]
                    # acc, avg_acc, cnt, pred
                    _, avg_acc, _, _ = accuracy(heatmap, scaled_targets)
                    loss += avg_acc
                else:
                    loss += self._compute_keypoint_loss(
                        heatmap, scaled_targets, image_size)

            loss /= len(heatmaps)
        return loss

    def _compute_keypoint_loss(
            self,
            heatmap,
            target,
            image_size) -> torch.Tensor:
        # target_weight = target[:, :, 1]
        target = target[:, :, :3]
        heatmap_gt_list, target_weight_list = [], []
        for i, joints in enumerate(target):
            heatmap_gt, target_weight = generate_target(joints, image_size)
            heatmap_gt_list.append(heatmap_gt.unsqueeze(dim=0))
            target_weight_list.append(target_weight.unsqueeze(dim=0))
            # print(heatmap_gt.shape, target_weight.shape)
        heatmap_gt = torch.cat(heatmap_gt_list, dim=0).to(heatmap.device)
        target_weight = torch.cat(target_weight_list, dim=0).to(heatmap.device)
        # print(heatmap.shape, heatmap_gt.shape, target_weight.shape)
        loss = self.criterion(heatmap, heatmap_gt, target_weight)
        return loss

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, 256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
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
                hidden_states=hs[keep],
                human_hidden_states=hs[keep_h]
            ))

        return region_props

    def postprocessing(
            self,
            boxes,
            bh,
            bo,
            logits,
            prior,
            objects,
            attn_maps,
            image_sizes):
        n = [len(b) for b in bh]
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, attn, size in zip(
            boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], attn_maps=attn, size=size
            ))

        return detections

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.detector.backbone(images)

        src, mask = features[-1].decompose()
        assert mask is not None
        input_proj = self.detector.input_proj(src)
        hs = self.detector.transformer(
            input_proj, mask, self.detector.query_embed.weight, pos[-1])[0]

        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        results = {
            'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results, hs[-1])

        pose_heatmaps = []
        for region_prop in region_props:
            human_hidden_states = region_prop["human_hidden_states"]
            pose_heatmap = self.pose_net(human_hidden_states)
            pose_heatmaps.append(pose_heatmap)

        logits, prior, bh, bo, objects, attn_maps = self.interaction_head(
            features[-1].tensors, image_sizes, region_props
        )

        boxes = [r['boxes'] for r in region_props]

        # for i, _ in enumerate(targets):
        #     targets[i]['size'] = image_sizes[i]

        if self.training:
            pose_loss = self.compute_keypoint_loss(
                pose_heatmaps, targets, image_sizes)
            interaction_loss = self.compute_interaction_loss(
                boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(
                interaction_loss=interaction_loss,
                pose_loss=pose_loss
            )
            return loss_dict

        detections = self.postprocessing(
            boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes)
        return detections


def build_detector(config, args, class_corr):
    detr, _, postprocessors = build_model(config, args)
    if os.path.exists(config.MODEL.PRETRAINED):
        if not config.DDP or dist.get_rank() == 0:
            print(
                f"Load weights for the object detector from {config.MODEL.PRETRAINED}")
        detr.load_state_dict(
            torch.load(
                config.MODEL.PRETRAINED,
                map_location='cpu')['model_state_dict'])
    predictor = torch.nn.Linear(
        config.MODEL.REPR_DIM * 2,
        config.DATASET.NUM_CLASSES)
    interaction_head = InteractionHead(
        predictor, config.MODEL.HIDDEN_DIM, config.MODEL.REPR_DIM,
        detr.backbone[0].num_channels,
        config.DATASET.NUM_CLASSES, config.HUMAN_ID, class_corr
    )
    pose_net = get_post_net_without_res(config, args)
    detector = UPT(
        detr, postprocessors['bbox'], interaction_head,
        human_idx=config.HUMAN_ID, num_classes=config.DATASET.NUM_CLASSES,
        pose_net=pose_net, alpha=config.ALPHA, gamma=config.GAMMA,
        box_score_thresh=config.BOX_SCORE_THRESH,
        fg_iou_thresh=config.FG_IOU_THRESH,
        min_instances=config.MIN_INSTANCES,
        max_instances=config.MAX_INSTANCES,
    )
    return detector
