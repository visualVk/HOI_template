import argparse

from ipywidgets.widgets import interaction

from model.ds import nested_tensor_from_tensor_list
from model.interaction_head import InteractionHead
from config import config
from model.detr import build_detr
import torch


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="HOI Transformer", add_help=False)
    parser.add_argument("--local_rank", default=0,
                        type=int, help="Local rank")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("--world_size", default=0, type=int, help="World size")
    parser.add_argument("-nr", "--nr", default=0, type=int,
                        help="ranking within the nodes")
    parser.add_argument(
        "--backend",
        default='gloo',
        type=str,
        help="ddp backend")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./config.yaml",
        help="config written by yaml path")

    return parser


if __name__ == '__main__':
    parser = get_parse_args()
    args = parser.parse_args()
    detr, criterion, postprocessors = build_detr(config, args)
    image = torch.randn((3, 224, 224))
    image_shape = torch.tensor([224, 224]).view(1, 2)
    nested_tensor = nested_tensor_from_tensor_list([image])
    interaction_head = InteractionHead(detr)
    result = interaction_head(nested_tensor, image_shape)
    print(result)
