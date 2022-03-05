from model.gcn import GCN
from torch import nn
import torch

if __name__ == '__main__':
    gcn = GCN(1, 1024)
    node = torch.randn((2, 10, 1024))
    labels = torch.cat([torch.ones((2, 4)), torch.zeros((2, 6))], dim=-1)
    coords = torch.randn((2, 10, 4))
    image_shapes = (224, 224)
    node_list, human_node_list, unary_list = [], [], []
    for node_i, coords_i, labels_i in zip(node, coords, labels):
        node, human_node, unary, x_keep, y_keep = gcn(
            labels_i, node_i, coords_i, image_shapes)
        print(node.shape, human_node.shape, unary.shape)
        node_list.append(node)
        human_node_list.append(human_node)
        unary_list.append(unary)
