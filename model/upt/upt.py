from model.pose.AdaptivePose import build_simple_joint
from utils.logger import log_every_n
from utils.misc import nested_tensor_from_tensor_list
from utils import box_ops
from model.detr.detr import build_detr as build_model
from model.simple_baseline import get_post_net_without_res
from model.lpn import build_lpn_model, LPN_2
from loss.simplebaseline_criterion import JointsMSELoss, JointsL1Loss
from scipy.optimize import linear_sum_assignment
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from utils.keypoint import accuracy, generate_target
from typing import Optional, List, Tuple, Union, Dict
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
                 device: torch.device, train_pose: bool,
                 idx_map: Optional[torch.Tensor] = None,
                 interaction_head_of_pose: Optional[nn.Module] = None,
                 pose_net: Optional[nn.Module] = None,
                 lpn: Optional[nn.Module] = None,
                 alpha: float = 0.5, gamma: float = 2.0,
                 box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
                 min_instances: int = 3, max_instances: int = 15,
                 ) -> None:
        super().__init__()
        self.device = device
        self.train_pose = train_pose
        self.detector = detector
        self.pose_net = pose_net
        self.lpn = lpn

        self.postprocessor = postprocessor
        self.interaction_head = interaction_head
        self.interaction_head_of_pose = interaction_head_of_pose
        self.criterion = JointsL1Loss(use_target_weight=False)

        self.human_idx = human_idx
        self.num_classes = num_classes
        self.idx_map = idx_map

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
            n_p = torch.as_tensor([n_p], device=labels.device)
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )
        # return loss / n_p
        return loss / (n_p + 1 if n_p == 0 else n_p)

    def associate_joints_with_ground_truth(self, boxes_h, boxes_o, targets):
        joints_gt = targets["keypoints"]
        n = boxes_h.shape[0]

        gt_bx_h = targets['boxes_h']
        gt_bx_o = targets['boxes_o']

        x, y = torch.nonzero(box_iou(boxes_h, gt_bx_h) >=
                             self.fg_iou_thresh).unbind(1)
        return x, y

    def compute_keypoinit_acc(
            self,
            boxes,
            bh,
            bo,
            images_shapes,
            preds,
            targets):
        acc = 0
        l1loss = nn.MSELoss()
        tot_p = 0
        for pred, bx, h, o, sizes, target in zip(
                preds, boxes, bh, bo, images_shapes, targets):
            s_h, s_w = sizes
            joints_gt = target["keypoints"][:, :, :-1]
            pred = pred[h]
            joints_gt /= torch.tensor([s_h, s_w]).to(pred.device)
            x, y = self.associate_joints_with_ground_truth(
                bx[h], bx[o], target)
            pred = pred[x]
            joints_gt = joints_gt[y]
            thresh = nn.Threshold(0, 0)
            acc += thresh(l1loss(pred, joints_gt)) / joints_gt.size(1)
            tot_p += x.size(0)
        acc /= len(preds)
        return acc

    def prepare_region_proposals(self, results, hidden_states):
        # hidden_states: [num_queries, hidden_size]->[100, 256]
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
                human_hidden_states=hs[keep_h],
                object_hidden_states=hs[keep_o]
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
        hs, memory = self.detector.transformer(
            input_proj, mask, self.detector.query_embed.weight, pos[-1])

        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        results = {
            'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results, hs[-1])

        if self.pose_net is not None:
            pose_reprs = []
            pose_joints = []
            for i, region_prop in enumerate(region_props):
                human_hidden_states = region_prop["human_hidden_states"]
                object_hidden_states = region_prop["object_hidden_states"]
                pose_repr, joint_coord = self.pose_net(human_hidden_states)

                joint_coord = torch.clamp(joint_coord, torch.tensor([0, 0], device=self.device), image_sizes[i])
                # pose_repr_add = torch.zeros_like(region_prop["hidden_states"]).to(human_hidden_states.device)
                pose_repr = torch.cat([pose_repr, object_hidden_states])
                pose_reprs.append(pose_repr)
                pose_joints.append(joint_coord)

        logits, prior, bh, bo, objects, attn_maps = self.interaction_head(
            features[-1].tensors, image_sizes, region_props
        )

        boxes = [r['boxes'] for r in region_props]
        if self.lpn is not None:
            for i, region_prop in enumerate(region_props):
                region_props[i]["hidden_states"] = pose_reprs[i]
            logits_p, prior_p, _, _, _, attn_maps = self.interaction_head(
                features[-1].tensors, image_sizes, region_props
            )

        # for i, _ in enumerate(targets):
        #     targets[i]['size'] = image_sizes[i]

        if self.training:
            interaction_loss = self.compute_interaction_loss(
                boxes,
                bh,
                bo,
                logits,
                prior,
                targets) if not self.train_pose else torch.tensor(
                [0],
                device=self.device)
            pose_loss = self.compute_keypoinit_acc(
                boxes, bh, bo, image_sizes, pose_joints, targets)

            interaction_part_loss = self.compute_interaction_loss(
                boxes,
                bh,
                bo,
                logits_p,
                prior_p,
                targets) if not self.train_pose else torch.tensor(
                [0],
                device=self.device)
            loss_dict = dict(
                interaction_loss=interaction_loss,
                interaction_part_loss=interaction_part_loss,
                pose_loss=pose_loss)
            return loss_dict

        if self.train_pose:
            pose_joints = [pose_joint * image_sizes[i] for i, pose_joint in enumerate(pose_joints)]
            return pose_joints

        detections = self.postprocessing(
            boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes)
        return detections


def build_detector(config, args, class_corr, idx_map=None):
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
    use_pose = config.POSE_NET
    if use_pose:
        # pose_net = get_post_net_without_res(config, args)
        pose_net = build_simple_joint(
            config.MODEL.HIDDEN_DIM,
            config.MODEL.NUM_JOINTS)
    else:
        pose_net = None
    use_lpn = config.LPN
    if use_lpn:
        lpn = build_lpn_model(
            config.MODEL.HIDDEN_DIM,
            config.MODEL.NUM_QUERIES,
            config.MODEL.NUM_JOINTS,
            config.MODEL.EXTRA.HEATMAP_SIZE)
        # lpn = LPN_2()
    else:
        lpn = None
    detector = UPT(
        detr,
        postprocessors['bbox'],
        interaction_head,
        human_idx=config.HUMAN_ID,
        num_classes=config.DATASET.NUM_CLASSES,
        device=args.local_rank,
        train_pose=config.TRAIN_POSE_NET,
        idx_map=idx_map,
        pose_net=pose_net,
        lpn=lpn,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        box_score_thresh=config.BOX_SCORE_THRESH,
        fg_iou_thresh=config.FG_IOU_THRESH,
        min_instances=config.MIN_INSTANCES,
        max_instances=config.MAX_INSTANCES,
    )
    return detector


def remove_negative(results: List[Dict[str, Tensor]], idx_map: Dict[int, int]):
    for i, result in enumerate(results):
        scores = result["scores"]
        boxes = result["boxes"]
        labels = result["labels"]
        selected_index = torch.zeros_like(labels, dtype=torch.bool)
        for j, label in enumerate(labels):
            if label.item() in idx_map.keys():
                selected_index[j] = True
                labels[j] = idx_map[label.item()]
        scores = scores[selected_index]
        boxes = boxes[selected_index]
        labels = labels[selected_index]
        results[i]["scores"] = scores
        results[i]["boxes"] = boxes
        results[i]["labels"] = labels
    # return results
