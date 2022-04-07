import argparse
import os

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.upt_vcoco_config import config as cfg
from dataset.data_factory import DataFactory
from dataset.data_factory import custom_collate
from model.upt.upt import build_detector
from utils import misc
from utils.draw_tensorboard import TensorWriter
from utils.model import adapt_device
from utils.vsrl_eval import VCOCOeval


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="HOI Transformer", add_help=False)
    parser.add_argument("--local_rank", default=0,
                        type=int, help="Local rank")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("--world_size", default=1, type=int, help="World size")
    parser.add_argument("-nr", "--nr", default=0, type=int,
                        help="ranking within the nodes")
    parser.add_argument(
        '--partitions',
        nargs='+',
        default=[
            'trainval',
            'test'],
        type=str)
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


def preprocess_config(args: argparse.Namespace):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if cfg.DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])


def main(rank, args, config):
    # whether to use DDP
    args.local_rank = rank
    if config.DDP and config.CUDNN.ENABLED and torch.cuda.is_available():
        misc.init_distributed_mode(args)

    misc.fix_random_seed(args, config)

    data_root = config.DATASET.ROOT
    testset = DataFactory(
        name='vcoco',
        partition=args.partitions[1],
        data_root=data_root)

    begin_epoch = config.TEST.BEGIN_EPOCH
    end_epoch = config.TEST.END_EPOCH
    writer = TensorWriter().writer

    vsrl_annot_file = "data/mscoco2014/vcoco_test.json"
    coco_file = "data/mscoco2014/instances_vcoco_all_2014.json"
    split_file = "data/mscoco2014/splits/vcoco_test.ids"
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    for epoch in range(begin_epoch, end_epoch):
        print(f"evaluate epoch {epoch}:")
        # Change this line to match the path of your cached file
        det_file = f"./data/cache/cache_{epoch}.pkl"

        print(f"Loading cached results from {det_file}.")
        mAP_a, mAP_r_1, mAP_r_2 = vcocoeval._do_eval(det_file, ovr_thresh=0.5)
        # if misc.is_main_process():
        writer.add_scalar("mAP of agent", mAP_a, epoch)
        writer.add_scalar("mAP of role in scenario 1", mAP_r_1, epoch)
        writer.add_scalar("mAP of role in scenario 2", mAP_r_2, epoch)


if __name__ == '__main__':
    parser = get_parse_args()
    args = parser.parse_args()
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "8888"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    print(args)
    main(0, args, cfg)
    # mp.spawn(main, nprocs=args.world_size, args=(args, cfg,))
