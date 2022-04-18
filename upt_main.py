import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from utils.logger import setup_logger, log_every_n
from dataset.data_factory import DataFactory
from model.upt.upt import build_detector
from dataset.data_factory import custom_collate
from config.upt_vcoco_config import config, update_config_by_yaml
from engine.upt_engine import build_upt_engine
from utils import misc

# torch.autograd.set_detect_anomaly(True)
from utils.draw_tensorboard import TensorWriter


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
        "--cache_dir",
        default="data/cache",
        type=str,
        help="cache directory")
    parser.add_argument("--port", default="8888", type=str)
    parser.add_argument(
        "--pretrained_path",
        default="data/upt-r50-vcoco.pt",
        type=str,
        help="model pretrained path")
    parser.add_argument("--cache_epoch", default=19, type=int)
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

    log_every_n(20, "loaded data loader")
    if config.DATASET.NAME == "hicodet":
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_classes = 117
    elif config.DATASET.NAME == "mscoco2014":
        object_to_target = list(
            train_loader.dataset.dataset.object_to_action.values())
    upt = build_detector(config, args, object_to_target)
    upt_trainer = build_upt_engine(upt, config, args)

    upt_trainer.update_train_dataloader(train_loader)
    upt_trainer.update_val_dataloader(test_loader)

    if config.IS_TRAIN:
        upt_trainer.train(evaluate=False)

    if config.CACHE:
        net_state_dict = torch.load(
            args.pretrained_path,
            map_location="cpu")
        upt = build_detector(config, args, object_to_target)
        upt.load_state_dict(net_state_dict["model_state_dict"])
        print("loaded pretrained model")
        upt_trainer.reload_eval_model_in_epoch(upt)
        upt_trainer.cache_vcoco(args.cache_epoch, args.cache_dir)

    # else:
    #     upt_trainer.update_test_dataloader(test_loader)
    #     upt_trainer.eval()


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
