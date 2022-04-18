import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from utils.logger import setup_logger, log_every_n
from utils.model import with_ddp, load_checkpoint
from utils.dataloader import get_dataset
from config.asnet_hico_config import config, update_config_by_yaml
from dataset.hico_det import collect

from engine.asnet_engine import ASNetEngine
from utils import misc
from model.asnet import build_model

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
    parser.add_argument("--port", default="8888", type=str)
    parser.add_argument(
        "--backend",
        default='nccl',
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

    device = torch.device(rank)
    model, criterion, postprocessors = build_model(config, device)
    model.to(device)
    model = with_ddp(model, rank)
    train_dataset, eval_dataset = get_dataset(config)
    batch_size = config.DATASET.IMG_NUM_PER_GPU
    model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "rel" in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "rel" not in n and p.requires_grad],
            "lr": config.TRAIN.LR_BACKBONE,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.TRAIN.LR,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.TRAIN.LR_DROP)
    model, optimizer, lr_scheduler, last_iter = load_checkpoint(
        config, model, optimizer, lr_scheduler, device)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=False,
        shuffle=(train_sampler is None),
        drop_last=True,
        collate_fn=collect,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collect,
        num_workers=config.WORKERS
    )

    engine = ASNetEngine(
        model,
        args,
        config,
        postprocessors=postprocessors,
        device="cuda",
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_norm=config.TRAIN.CLIP_MAX_NORM)
    engine.update_train_dataloader(train_loader)
    engine.update_val_dataloader(eval_loader)

    if config.EVAL:
        engine.evaluate()
        return
    engine.train(evaluate=False)


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
