import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import box_ops
from utils.misc import accuracy, is_dist_avail_and_initialized, get_world_size

num_humans = 2


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, num_actions, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # 91
        self.num_actions = num_actions
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # why num_humans plus 1: maybe you will predict no human or empty set, so, in fact, we will 3 classes: foreground, background and no object
        human_empty_weight = torch.ones(num_humans + 1)
        human_empty_weight[-1] = self.eos_coef
        self.register_buffer('human_empty_weight', human_empty_weight)

        object_empty_weight = torch.ones(num_classes + 1)
        object_empty_weight[-1] = self.eos_coef
        self.register_buffer('object_empty_weight', object_empty_weight)

        action_empty_weight = torch.ones(num_actions + 1)
        action_empty_weight[-1] = self.eos_coef
        self.register_buffer('action_empty_weight', action_empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'human_pred_logits' in outputs
        assert 'object_pred_logits' in outputs
        assert 'action_pred_logits' in outputs
        human_src_logits = outputs['human_pred_logits']
        object_src_logits = outputs['object_pred_logits']
        action_src_logits = outputs['action_pred_logits']

        idx = self._get_src_permutation_idx(indices)

        human_target_classes_o = torch.cat(
            [t["human_labels"][J] for t, (_, J) in zip(targets, indices)])
        object_target_classes_o = torch.cat(
            [t["object_labels"][J] for t, (_, J) in zip(targets, indices)])
        action_target_classes_o = torch.cat(
            [t["action_labels"][J] for t, (_, J) in zip(targets, indices)])

        human_target_classes = torch.full(human_src_logits.shape[:2], num_humans,
                                          dtype=torch.int64, device=human_src_logits.device)
        human_target_classes[idx] = human_target_classes_o

        object_target_classes = torch.full(object_src_logits.shape[:2], self.num_classes,
                                           dtype=torch.int64, device=object_src_logits.device)
        object_target_classes[idx] = object_target_classes_o

        action_target_classes = torch.full(action_src_logits.shape[:2], self.num_actions,
                                           dtype=torch.int64, device=action_src_logits.device)
        action_target_classes[idx] = action_target_classes_o

        human_loss_ce = F.cross_entropy(human_src_logits.transpose(1, 2),
                                        human_target_classes, self.human_empty_weight)
        object_loss_ce = F.cross_entropy(object_src_logits.transpose(1, 2),
                                         object_target_classes, self.object_empty_weight)
        action_loss_ce = F.cross_entropy(action_src_logits.transpose(1, 2),
                                         action_target_classes, self.action_empty_weight)
        loss_ce = human_loss_ce + object_loss_ce + 2 * action_loss_ce
        losses = {
            'loss_ce': loss_ce,
            'human_loss_ce': human_loss_ce,
            'object_loss_ce': object_loss_ce,
            'action_loss_ce': action_loss_ce
        }

        if log:
            losses['class_error'] = 100 - \
                accuracy(action_src_logits[idx], action_target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['action_pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["action_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'human_pred_boxes' in outputs
        assert 'object_pred_boxes' in outputs
        # indice: calculated by minimum match loss which use matcher to calculate
        idx = self._get_src_permutation_idx(indices)

        human_src_boxes = outputs['human_pred_boxes'][idx]
        human_target_boxes = torch.cat(
            [t['human_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        object_src_boxes = outputs['object_pred_boxes'][idx]
        object_target_boxes = torch.cat(
            [t['object_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        human_loss_bbox = F.l1_loss(
            human_src_boxes, human_target_boxes, reduction='none')
        object_loss_bbox = F.l1_loss(
            object_src_boxes, object_target_boxes, reduction='none')

        losses = dict()
        losses['human_loss_bbox'] = human_loss_bbox.sum() / num_boxes
        losses['object_loss_bbox'] = object_loss_bbox.sum() / num_boxes
        losses['loss_bbox'] = losses['human_loss_bbox'] + \
            losses['object_loss_bbox']

        human_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(human_src_boxes),
            box_ops.box_cxcywh_to_xyxy(human_target_boxes)))
        object_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(object_src_boxes),
            box_ops.box_cxcywh_to_xyxy(object_target_boxes)))
        losses['human_loss_giou'] = human_loss_giou.sum() / num_boxes
        losses['object_loss_giou'] = object_loss_giou.sum() / num_boxes

        losses['loss_giou'] = losses['human_loss_giou'] + \
            losses['object_loss_giou']
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["human_labels"]) for t in targets)

        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
