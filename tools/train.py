# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
from bisect import bisect_right
from torch.backends import cudnn

import sys
sys.path.append(".")
from config import cfg
from data import get_data_bunch
from engine.trainer import do_train
from fastai.vision import *
from layers import reidLoss
from modeling import build_model
from solver import *
from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    data_bunch, test_labels, num_query = get_data_bunch(cfg)

    # prepare model
    model = build_model(cfg, data_bunch.c)

    if cfg.SOLVER.OPT == 'adam':    opt_fns = partial(torch.optim.Adam)
    elif cfg.SOLVER.OPT == 'sgd': opt_fns = partial(torch.optim.SGD, momentum=0.9)
    else:                           raise NameError(f'optimizer {cfg.SOLVER.OPT} not support')

    def lr_multistep(start: float, end: float, pct: float):
        warmup_factor = 1
        gamma = cfg.SOLVER.GAMMA
        milestones = [1.0 * s / cfg.SOLVER.MAX_EPOCHS for s in cfg.SOLVER.STEPS]
        warmup_iter = 1.0 * cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_EPOCHS
        if pct < warmup_iter:
            alpha = pct / warmup_iter
            warmup_factor = cfg.SOLVER.WARMUP_FACTOR * (1 - alpha) + alpha
        return start * warmup_factor * gamma ** bisect_right(milestones, pct)

    lr_sched = Scheduler(cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_EPOCHS, lr_multistep)

    loss_func = reidLoss(cfg.SOLVER.LOSSTYPE, cfg.SOLVER.MARGIN)

    do_train(
        cfg,
        model,
        data_bunch,
        test_labels,
        opt_fns,
        lr_sched,
        loss_func,
        num_query,
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument('-cfg',
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR): os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR, 0)
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
