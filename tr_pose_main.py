import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from utils.logger import setup_logger, log_every_n
from dataset.data_factory import DataFactory
from model.upt.TRPose import build_tr_pose
from dataset.data_factory import custom_collate
from config.upt_vcoco_config import config, update_config_by_yaml
from engine.upt_engine import build_upt_engine
from utils import misc
from loguru import logger
from utils.draw_tensorboard import TensorWriter
from loss.simplebaseline_criterion import JointsMSELoss
from engine.tr_pose_engine import build_tr_pose_engine
from utils import keypoint


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
        default="./upt_config.yaml",
        help="config written by yaml path")

    return parser


def main(rank, args, config):
    # whether to use DDP
    args.local_rank = rank
    if config.CUDNN.ENABLED and torch.cuda.is_available():
        misc.init_distributed_mode(args)

    misc.fix_random_seed(args, config)

    if misc.is_main_process():
        TensorWriter(config)

    log_file = os.path.join(config.LOG_DIR, f"{config.PROJECT_NAME}.log")
    setup_logger(log_file, args.local_rank)

    data_root = config.DATASET.ROOT

    trainset = DataFactory(
        name='vcoco',
        partition=args.partitions[0],
        data_root=data_root)
    testset = DataFactory(
        name='vcoco',
        partition=args.partitions[1],
        data_root=data_root)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.WORKERS, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=0)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=config.WORKERS, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    logger.info("==> loaded data loader")
    if config.DATASET.NAME == "hicodet":
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_classes = 117
    elif config.DATASET.NAME == "mscoco2014":
        object_to_target = list(
            train_loader.dataset.dataset.object_to_action.values())

    model = build_tr_pose(config, args)
    criterion = JointsMSELoss()
    acc = keypoint.accuracy
    tr_pose_engine = build_tr_pose_engine(model, criterion, acc, config, args)
    if not config.IS_TRAIN:
        return
    tr_pose_engine.train()


if __name__ == '__main__':
    parser = get_parse_args()
    args = parser.parse_args()
    # update_config_by_yaml(args.path)
    cfg = config
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    print(args)
    mp.spawn(main, nprocs=args.world_size, args=(args, cfg,))
