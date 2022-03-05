import argparse

import torch
from config import config
from model.ds import NestedTensor, nested_tensor_from_tensor_list
from model.detr import build_detr


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
    d = torch.load('./data/detr-r50.pth')
    print(d['model'].keys())
    # image = torch.randn((3, 224, 224))
    # nested_tensor = nested_tensor_from_tensor_list([image])
    # detr(nested_tensor)
