# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

import warnings
import torch
from torch.backends import cudnn

sys.path.append(".")
from config import cfg
from data import get_dataloader
from engine.trainer import do_train
from utils.logger import setup_logger


def train(cfg, local_rank):
    # prepare dataset
    tng_loader, val_loader, num_classes, num_query = get_dataloader(cfg)

    do_train(
        cfg,
        local_rank,
        tng_loader,
        val_loader,
        num_classes,
        num_query,
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        '-cfg', "--config_file", 
        default="", 
        metavar="FILE", 
        help="path to config file", 
        type=str
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.SOLVER.DIST = num_gpus > 1

    if cfg.SOLVER.DIST:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        torch.cuda.synchronize()

    cfg.freeze()

    log_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST_NAMES, 'version_'+cfg.MODEL.VERSION)
    if not os.path.exists(log_save_dir): os.makedirs(log_save_dir)

    logger = setup_logger("reid_baseline", log_save_dir, 0)
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    train(cfg, args.local_rank)


if __name__ == '__main__':
    main()
