import argparse
import os

from config.upt_vcoco_config import config as cfg
from utils.logger import setup_logger, create_small_table, log_every_n
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
        "--cache_file",
        default="data/cache/cache_19.pkl",
        type=str)
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

    vsrl_annot_file = "data/mscoco2014/vcoco_test.json"
    coco_file = "data/mscoco2014/instances_vcoco_all_2014.json"
    split_file = "data/mscoco2014/splits/vcoco_test.ids"
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    # for epoch in range(begin_epoch, end_epoch):
    print(f"evaluate {args.cache_file}")
    # Change this line to match the path of your cached file
    det_file = args.cache_file

    print(f"Loading cached results from {det_file}.")
    agent_ap, s1_ap, s2_ap = vcocoeval._do_eval(det_file, ovr_thresh=0.5)
    ap_table = create_small_table(
        dict(
            agent_ap=agent_ap,
            scenario_1_ap=s1_ap,
            scenario_2_ap=s2_ap))
    log_every_n(20, f"#{ap_table}")


if __name__ == '__main__':
    parser = get_parse_args()
    args = parser.parse_args()
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "8888"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    print(args)
    log_dir = os.path.join(cfg.LOG_DIR, "log_upt_inference.log")
    setup_logger(log_dir)
    main(0, args, cfg)
    # mp.spawn(main, nprocs=args.world_size, args=(args, cfg,))
