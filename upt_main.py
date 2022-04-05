import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from dataset.data_factory import DataFactory
from model.upt.upt import build_detector
from dataset.data_factory import custom_collate
from config.upt_vcoco_config import config, update_config_by_yaml
from engine.upt_engine import build_upt_engine
from utils import misc

# torch.autograd.set_detect_anomaly(True)


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
    if config.DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])


def main(rank, args, config):
    # whether to use DDP
    args.local_rank = rank
    if config.DDP and config.CUDNN.ENABLED and torch.cuda.is_available():
        misc.init_distributed_mode(args)

    misc.fix_random_seed(args, config)

    # preprocess_config(args)

    # data_root = os.path.join(config.DATASET.ROOT, config.DATASET.NAME)
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

    # args.human_idx = 0
    # if args.dataset == 'hicodet':
    #     object_to_target = train_loader.dataset.dataset.object_to_verb
    #     args.num_classes = 117
    # elif args.dataset == 'vcoco':
    object_to_target = list(
        train_loader.dataset.dataset.object_to_action.values())
    upt = build_detector(config, args, object_to_target)

    upt_trainer = build_upt_engine(upt, config, args)
    if config.MODEL_TYPE == "train":
        upt_trainer.update_train_dataloader(train_loader)
        upt_trainer.update_val_dataloader(train_loader)
        upt_trainer.train()
    else:
        upt_trainer.update_test_dataloader(test_loader)
        if config.TEST.SAVE:
            cache_dir = "./data/cache"
            upt_trainer.cache_vcoco(19, cache_dir)
        else:
            # upt = build_detector(config, args, object_to_target)
            # upt_trainer.update_attr('model', upt)
            upt_trainer.eval()


if __name__ == '__main__':
    parser = get_parse_args()
    args = parser.parse_args()
    cfg = config
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    print(args)
    mp.spawn(main, nprocs=args.world_size, args=(args, cfg,))
