from model.STIGPN import VisualModelV, SemanticModelV
import argparse
import torch


parser = argparse.ArgumentParser(description="You Can Do It!")
parser.add_argument('--model', default='VisualModelV',
                    help='VisualModelV,SemanticModelV')
parser.add_argument('--task', default='Detection')
parser.add_argument(
    '--batch_size',
    '--b_s',
    type=int,
    default=3,
    help='batch size: 1')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='number of beginning epochs : 0')
parser.add_argument('--epoch', type=int, default=300,
                    help='number of epochs to train: 300')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate: 0.0001')  # 2e-5
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.8,
    help='learning rate: 0.0001')
parser.add_argument(
    '--nr_boxes',
    type=int,
    default=6,
    help='number of bbox : 6')
parser.add_argument(
    '--nr_frames',
    type=int,
    default=10,
    help='number of frames : 10')
parser.add_argument(
    '--subact_classes',
    type=int,
    default=10,
    help='number of subact_classes : 10')
parser.add_argument(
    '--afford_classes',
    type=int,
    default=12,
    help='number of afford_classes : 12')
parser.add_argument(
    '--feat_drop',
    type=float,
    default=0,
    help='dropout parameter: 0')
parser.add_argument(
    '--attn_drop',
    type=float,
    default=0,
    help='dropout parameter: 0')
parser.add_argument(
    '--cls_dropout',
    type=float,
    default=0,
    help='dropout parameter: 0')
parser.add_argument('--step_size', type=int, default=50,
                    help='number of steps for validation loss: 10')
parser.add_argument('--eval_every', type=int, default=1,
                    help='number of steps for validation loss: 10')
parser.add_argument(
    '--obj_scal',
    type=int,
    default=1,
    help='number of steps for validation loss: 10')

if __name__ == '__main__':
    # num_objs,
    # node_features,
    # box_input,
    # box_categories,
    args = parser.parse_args()
    device = torch.device('cuda')
    num_objs = torch.randint(1, 6, (1, 1), device=device)
    node_features = torch.randn((1, args.nr_frames, 6, 2048), device=device)
    box_input = torch.randn((1, args.nr_frames, 6, 4), device=device)
    box_categories = torch.ones((1, args.nr_frames, 6), device=device)
    model = VisualModelV(args).to(device)
    output = model(num_objs, node_features, box_input, box_categories)
