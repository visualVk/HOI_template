import torch
from torch import nn

from model.ds import nested_tensor_from_tensor_list


class P2PNet(nn.Module):
    def __init__(
            self,
            detector: nn.Module,
            posenet: nn.Module,
            num_parts: int):
        super(P2PNet, self).__init__()
        self.detector = detector
        self.posenet = posenet

        self.q_linear = nn.Linear(8 * 8, 16 * 16)
        self.k_linear = nn.Linear(8 * 8, 16 * 16)
        self.v_linear = nn.Linear(8 * 8, 16 * 16)
        self.atten = nn.MultiheadAttention(
            2048 * 8 * 8, 2, device=detector.device)
        self.hm_pt = nn.Conv1d(2048, num_parts * 5, (1, 1))

        self.num_parts = num_parts

    def forward(self, samples):
        bs = samples.size(0)
        image_shapes = torch.tensor(
            [im.shape[-2:] for im in samples], device=samples.device).view(bs, 2)

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(
                [image for image in samples])

        # >>>>>>>>>>>>>>>>>>>> transformer <<<<<<<<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        input_proj = self.detr.input_proj(src)
        # hs: [enc_layers, bs, c, hidden_states], [x, bs, c, 256]
        hs = self.detr.transformer(input_proj, mask,
                                   self.detr.query_embed.weight, pos[-1])[0]
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()

        # hs_last = hs[-1]
        # hm_pt = self.hm_pt(hs_last).view(
        #     bs, self.num_parts, 5, 256)  # hm_pt: [bs, num_parts, 5, 256]
        #
        # pose_feature, deconv_feature = self.posenet(samples)
        # pose_feature = pose_feature.view(bs, -1, 8 * 8).permute(1, 0, 2)
        # k = self.k_linear(pose_feature)
        # q = self.q_linear(pose_feature)
        # v = self.v_linear(pose_feature)
        # v = self.atten(q, k, v).permute(1, 0, 2).view(bs, -1, 256)
