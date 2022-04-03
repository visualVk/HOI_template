from typing import Tuple, List, Optional, OrderedDict, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from utils import box_ops
from utils.ops import compute_spatial_encodings, binary_focal_loss
from dgl.nn.pytorch.conv import GATConv

human_id = 0


class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """

    def __init__(self,
                 appearance_size: int, spatial_size: int,
                 representation_size: int, cardinality: int
                 ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])

    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))


class MessageMBF(MultiBranchFusion):
    """
    MBF for the computation of anisotropic messages

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    node_type: str
        Nature of the sending node. Choose between `human` amd `object`
    cardinality: int
        The number of homogeneous branches
    """

    def __init__(self,
                 appearance_size: int,
                 spatial_size: int,
                 representation_size: int,
                 node_type: str,
                 cardinality: int
                 ) -> None:
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'human':
            self._forward_method = self._forward_human_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_human_nodes(
            self,
            appearance: Tensor,
            spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(
            appearance) == n_h, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def _forward_object_nodes(
            self,
            appearance: Tensor,
            spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(
            appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_h, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def forward(self, *args) -> Tensor:
        return self._forward_method(*args)


class GraphHead(nn.Module):
    """
    Graphical model head

    Parameters:
    -----------
    output_channels: int
        Number of output channels of the backbone
    roi_pool_size: int
        Spatial resolution of the pooled output
    node_encoding_size: int
        Size of the node embeddings
    num_cls: int
        Number of targe classes
    human_idx: int
        The index of human/person class in all objects
    object_class_to_target_class: List[list]
        The mapping (potentially one-to-many) from objects to target classes
    fg_iou_thresh: float, default: 0.5
        The IoU threshold to identify a positive example
    num_iter: int, default 2
        Number of iterations of the message passing process
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 roi_pool_size: int,
                 node_encoding_size: int,
                 representation_size: int,
                 num_cls: int, human_idx: int,
                 object_class_to_target_class: List[list],
                 fg_iou_thresh: float = 0.5,
                 num_iter: int = 2
                 ) -> None:

        super().__init__()

        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.human_idx = human_idx
        # detection的object的index跟target的Object的index，此处：detection的object
        # index跟hico object index
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Box head to map RoI features to low dimensional
        self.box_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            nn.ReLU()
        )

        # Compute adjacency matrix
        # self.adjacency = nn.Linear(representation_size, 1)
        self.adjacency = nn.Conv1d(in_channels, in_channels, 1)

        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        self.obj_to_sub = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        self.attention_head_g = MultiBranchFusion(
            256, 1024,
            representation_size, cardinality=16
        )

    def associate_with_ground_truth(self,
                                    boxes_h: Tensor,
                                    boxes_o: Tensor,
                                    targets: List[dict]
                                    ) -> Tensor:
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self,
                             x: Tensor, y: Tensor,
                             scores: Tensor,
                             object_class: Tensor
                             ) -> Tensor:
        """
        Parameters:
        -----------
            x: Tensor[M]
                Indices of human boxes (paired)
            y: Tensor[M]
                Indices of object boxes (paired)
            scores: Tensor[N]
                Object detection scores (before pairing)
            object_class: Tensor[N]
                Object class indices (before pairing)
        """
        prior_h = torch.zeros(len(x), self.num_cls, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
                          for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def forward(self,
                features: OrderedDict, image_shapes: List[Tuple[int, int]],
                box_features: Tensor, box_coords: List[Tensor],
                box_labels: List[Tensor], box_scores: List[Tensor],
                targets: Optional[List[dict]] = None
                ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor]
    ]:
        """
        Parameters:
        -----------
            features: OrderedDict
                Feature maps returned by FPN
            box_features: Tensor
                (N, C, P, P) Pooled box features
            image_shapes: List[Tuple[int, int]]
                Image shapes, heights followed by widths
            box_coords: List[Tensor]
                Bounding box coordinates organised by images
            box_labels: List[Tensor]
                Bounding box object types organised by images
            box_scores: List[Tensor]
                Bounding box scores organised by images
            targets: List[dict]
                Interaction targets with the following keys
                `boxes_h`: Tensor[G, 4]
                `boxes_o`: Tensor[G, 4]
                `labels`: Tensor[G]

        Returns:
        --------
            all_box_pair_features: List[Tensor]
            all_boxes_h: List[Tensor]
            all_boxes_o: List[Tensor]
            all_object_class: List[Tensor]
            all_labels: List[Tensor]
            all_prior: List[Tensor]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        # feature['3']=[n,256,2,2]-flatten->[n,256x2x2]
        global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        box_features = self.box_head(box_features)

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]

        counter = 0
        all_boxes_h = []
        all_boxes_o = []
        all_object_class = []
        all_labels = []
        all_prior = []
        all_box_pair_features = []
        for b_idx, (coords, labels, scores) in enumerate(
                zip(box_coords, box_labels, box_scores)):
            n = num_boxes[b_idx]
            device = box_features.device

            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                all_box_pair_features.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                all_boxes_h.append(torch.zeros(0, 4, device=device))
                all_boxes_o.append(torch.zeros(0, 4, device=device))
                all_object_class.append(
                    torch.zeros(
                        0,
                        device=device,
                        dtype=torch.int64))
                all_prior.append(
                    torch.zeros(
                        2,
                        0,
                        self.num_cls,
                        device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            if not torch.all(labels[:n_h] == self.human_idx):
                raise ValueError(
                    "Human detections are not permuted to the top")

            node_encodings = box_features[counter: counter + n]
            # Duplicate human nodes
            h_node_encodings = node_encodings[:n_h]
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten()
            y = y.flatten()

            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            # Reshape the spatial features
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n_h, n, -1)

            adjacency_matrix = torch.ones(n_h, n, device=device)
            for _ in range(self.num_iter):
                # Compute weights of each edge
                weights = self.attention_head(
                    torch.cat([
                        h_node_encodings[x],
                        node_encodings[y]
                    ], 1),
                    box_pair_spatial
                )
                adjacency_matrix = self.adjacency(weights).reshape(n_h, n)

                # Update human nodes
                messages_to_h = F.relu(torch.sum(
                    adjacency_matrix.softmax(dim=1)[..., None] *
                    self.obj_to_sub(
                        node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                h_node_encodings = self.norm_h(
                    h_node_encodings + messages_to_h
                )

                # Update object nodes (including human nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        h_node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                node_encodings = self.norm_o(
                    node_encodings + messages_to_o
                )

            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )

            all_box_pair_features.append(torch.cat([
                self.attention_head(
                    torch.cat([
                        h_node_encodings[x_keep],
                        node_encodings[y_keep]
                    ], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep]
                ), self.attention_head_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep])
            ], dim=1))
            all_boxes_h.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            all_prior.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )

            counter += n

        return all_box_pair_features, all_boxes_h, all_boxes_o, \
            all_object_class, all_labels, all_prior


class GCN(nn.Module):
    def __init__(
            self,
            human_id: int,
            node_encoding_size: int,
            num_iter=2,
            representation_size=1024,
            mask_size=16,
            gama=0.9
    ):
        super(GCN, self).__init__()
        self.human_id = human_id
        self.gama = gama
        self.num_iter = 2
        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        self.obj_to_sub = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )
        self.adjacency = nn.Sequential(
            nn.Linear(representation_size, 1),
            nn.ReLU()
        )

    def _bn(self, in_channels):
        block = []
        block.append(nn.BatchNorm1d(in_channels))
        block.append(nn.ReLU())
        return nn.Sequential(*block)

    def _update_node(
            self,
            node: Tensor,
            human_node: Tensor,
            adjacency: Tensor):
        _human_node = torch.einsum('yx,xa->ya', adjacency, node)
        _node = torch.einsum('yx,xa->ya', adjacency.T, human_node)

        node = node * self.gama + (1. - self.gama) * _node
        node = self.norm_o(node)

        human_node = human_node * self.gama + (1. - self.gama) * _human_node
        human_node = self.norm_h(human_node)
        return node, human_node

    def forward(self,
                labels: Tensor,
                node: Tensor,
                coords: Tensor,
                image_shapes: Tuple[int,
                                    int],
                bbox_mask: Optional[Tensor] = None) -> Tuple[Tensor,
                                                             Tensor,
                                                             Tensor,
                                                             Tensor,
                                                             Tensor]:
        # feature: [n, v]
        # print(
        # f"node:{node.shape}, labels:{labels.shape}, coords:{coords.shape},
        # image_shapes:{image_shapes}")
        n = node.size(0)
        n_h = torch.nonzero(labels == self.human_id).size(0)
        x, y = torch.meshgrid([torch.arange(n_h), torch.arange(n)])
        x_keep, y_keep = torch.nonzero(x != y).unbind(1)
        # print(x_keep, y_keep)

        x = x.flatten()
        y = y.flatten()
        # Compute spatial features
        box_pair_spatial = compute_spatial_encodings(
            [coords[x]], [coords[y]], [image_shapes]
        )
        box_pair_spatial = self.spatial_head(box_pair_spatial)
        # Reshape the spatial features
        # box_pair_spatial_reshaped = box_pair_spatial.view(n_h, n, -1)
        human_node = node[:n_h]

        adjacency = None
        for i in range(self.num_iter):
            weight = self.attention_head(torch.cat(
                [human_node[x], node[y]], dim=1), box_pair_spatial)
            now_adjacency = self.adjacency(weight).view(n_h, n)
            # adjacency = adjacency * self.gama + \
            #     (1.0 - self.gama) * now_adjacency if adjacency is not None else now_adjacency
            adjacency = now_adjacency
            node, human_node = self._update_node(node, human_node, adjacency)

        return human_node[x_keep], node[y_keep], adjacency[x_keep,
                                                           y_keep], x_keep, y_keep


class unit_GAT(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            feat_drop=0,
            attn_drop=0,
            activation=None):
        super(unit_GAT, self).__init__()
        self.gat = GATConv(
            in_feats=in_channels,
            out_feats=out_channels,
            num_heads=1,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=True,
            activation=activation)

    def forward(self, graph, x):
        # batch_size,nodes,frame,feature_size = x.shape
        # y = x.reshape(batch_size*nodes*frame,feature_size)
        y = self.gat(graph, x)
        # y = y.reshape(batch_size,nodes,frame,-1)

        return y


class GAT(nn.Module):
    def __init__(
            self,
            in_channels,
            nhidden,
            out_channels,
            feat_drop=0,
            attn_drop=0,
            activation=None):
        super(GAT, self).__init__()

        # self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GAT(
            in_channels,
            out_channels,
            feat_drop,
            attn_drop,
            activation)

    def forward(self, graph, x):
        # x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x


