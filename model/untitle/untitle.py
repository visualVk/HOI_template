import torch
import torch.nn.functional as F
from torch import nn
from utils.ops import compute_rsc

from utils.misc import NestedTensor, nested_tensor_from_tensor_list


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(
                n, k) for n, k in zip(
                [input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class NNPT(nn.Module):
    def __init__(
            self,
            detector,
            interaction_transformer,
            interaction_encoder,
            interaction_decoder,
            hidden_dim,
            num_queries,
            num_classes,
            rel_num_classes,
            hoi_aux_loss=True,
            human_idx=0) -> None:
        super(NNPT, self).__init__()
        self.detector = detector
        self.interaction_transformer = interaction_transformer
        self.interaction_encoder = interaction_encoder
        self.interaction_decoder = interaction_decoder
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.hoi_aux_loss = hoi_aux_loss
        self.human_idx = human_idx

        self.h_pointer_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.o_pointer_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # SFG
        self.oa_candidate = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # interaction branch
        self.rel_class_embed = nn.Linear(hidden_dim, rel_num_classes)
        self.rel_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.rel_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.tau = 0.05

    def forward(self, samples: NestedTensor, targets):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        shapes = torch.cat([target["size"]
                           for target in targets]).to(samples.device)
        # backbone
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        input_proj = self.input_proj(src)
        hs = self.transformer(input_proj, mask,
                              self.detector.query_embed.weight, pos[-1])[0]
        instr_repr = hs[-1]

        rels_repr = self.interaction_encoder(
            input_proj, mask, self.rel_query_embed.weight, pos[-1])[0]  # [bs, queries_num, hidden_dim]

        # Support Feature Generator
        rels_repr = self.oa_candidate(rels_repr)
        o_class = self.detector.class_embed(hs)
        o_coord = self.detector.bbox_embed(hs).sigmoid()
        for batch_idx, (obj, box_o) in enumerate(zip(o_class, o_coord)):
            o_max_indices = obj.max(-1)[-1]
            human_indices = torch.nonzero(
                o_max_indices == self.human_idx).squeeze(1)
            box_h_selected = box_o[human_indices]
            box_o_selected = box_o
            n_h = box_h_selected.size(0)
            n_o = box_o_selected.size(0)
            x, y = torch.meshgrid(
                torch.arange(n_h).to(samples.device),
                torch.arange(n_o).to(samples.device)
            )

            x_keep, y_keep = torch.nonzero(
                torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten()
            y = y.flatten()
            oa_rsc = compute_rsc(box_h_selected[x], box_o_selected[y], shapes)
            # re sample x, y, w, h to create a heatmap

        h_pointer_repr = F.normalize(
            self.h_pointer_embed(rels_repr), p=2, dim=-1)
        o_pointer_repr = F.normalize(
            self.o_pointer_embed(rels_repr), p=2, dim=-1)
        outputs_h_idx = [
            torch.bmm(
                rel_repr,
                instr_repr.transpose(
                    1,
                    2)) /
            self.tau for rel_repr in h_pointer_repr]
        outputs_o_idx = [
            torch.bmm(
                rel_repr,
                instr_repr.transpose(
                    1,
                    2)) /
            self.tau for rel_repr in o_pointer_repr]

        rels_repr = rels_repr
        rels_class = self.rel_class_embed(rels_repr)
        rels_repr_aug = F.normalize(rels_repr + hs, dim=-1)
        rels_coord = self.rel_bbox_embed(rels_repr_aug).sigmoid()
        h_rel_scores = [torch.bmm(output_h_idx, o_class[i])
                        for i, output_h_idx in enumerate(outputs_h_idx)]
        o_rel_scores = [torch.bmm(output_o_idx, o_class[i])
                        for i, output_o_idx in enumerate(outputs_o_idx)]

        out = {
            "pred_logits": o_class[-1],
            "pred_boxes": o_coord[-1],
            "pred_hidx": outputs_h_idx[-1],
            "pred_oidx": outputs_o_idx[-1],
            "pred_rel_logits": rels_class[-1],
            "pred_rel_coord": rels_coord[-1],
            "h_rel_score": h_rel_scores[-1],
            "o_rel_score": o_rel_scores[-1]
        }

        if self.hoi_aux_loss:  # auxiliary loss
            out['hoi_aux_outputs'] = self._set_aux_loss_with_tgt(
                o_class,
                o_coord,
                outputs_h_idx,
                outputs_o_idx,
                rels_class,
            ) if self.return_obj_class else self._set_aux_loss(
                o_class,
                o_coord,
                outputs_h_idx,
                outputs_o_idx,
                rels_class)
        return out

    @torch.jit.unused
    def _set_aux_loss(
            self,
            outputs_class,
            outputs_coord,
            outputs_hidx,
            outputs_oidx,
            outputs_action,
            outputs_h_rel_score,
            outputs_o_rel_score
    ):
        return [{'pred_logits': a,
                 'pred_boxes': b,
                 'pred_hidx': c,
                 'pred_oidx': d,
                 'pred_rel_logits': e,
                 'pred_h_rel_score': f,
                 'pred_o_rel_score': g} for a,
                b,
                c,
                d,
                e,
                f,
                g in zip(outputs_class[-1:].repeat((outputs_action.shape[0],
                                                    1,
                                                    1,
                                                    1)),
                         outputs_coord[-1:].repeat((outputs_action.shape[0],
                                                    1,
                                                    1,
                                                    1)),
                         outputs_hidx[:-1],
                         outputs_oidx[:-1],
                         outputs_action[:-1],
                         outputs_h_rel_score[:-1],
                         outputs_o_rel_score[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(
            self,
            outputs_class,
            outputs_coord,
            outputs_hidx,
            outputs_oidx,
            outputs_action,
            outputs_tgt):
        return [{'pred_logits': a,
                 'pred_boxes': b,
                 'pred_hidx': c,
                 'pred_oidx': d,
                 'pred_rel_logits': e,
                 'pred_obj_logits': f} for a,
                b,
                c,
                d,
                e,
                f in zip(outputs_class[-1:].repeat((outputs_action.shape[0],
                                                    1,
                                                    1,
                                                    1)),
                         outputs_coord[-1:].repeat((outputs_action.shape[0],
                                                    1,
                                                    1,
                                                    1)),
                         outputs_hidx[:-1],
                         outputs_oidx[:-1],
                         outputs_action[:-1],
                         outputs_tgt[:-1])]
