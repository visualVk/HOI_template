from typing import List, Optional
import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F

from utils.misc import nested_tensor_from_tensor_list
from model.simple_baseline import get_pose_net
from model.detr.detr import build_detr


class TRPose(nn.Module):
    def __init__(self, backbone, pose_net, res_feat_size) -> None:
        super().__init__()
        self.backbone = backbone
        self.pose_net = pose_net
        self.res_feat_size = res_feat_size

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.detector.backbone(images)

        src, mask = features[-1].decompose()
        src = F.resize(src, self.res_feat_size)

        heatmaps = self.pose_net(src)

        return heatmaps


def build_tr_pose(config, args):
    detr, criterion, postprocessor = build_detr(config, args)
    pose_net = get_pose_net(config, args)
    if config.PRETRAINED_POSE_NET is not None:
        pose_net.load_state_dict(
            torch.load(
                config.PRETRAINED_POSE_NET,
                map_location='cpu'))
    extra = config.MODEL.EXTRA
    tr_pose = TRPose(detr, pose_net, extra.POSE_RESNET.FEAT_SIZE)
    return tr_pose
