# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import sys
from bisect import bisect_right

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import get_data_bunch
from engine.trainer import do_train
from layers import make_loss
from modeling import build_model
from utils.logger import Logger
from fastai.vision import *


def train(cfg):
    # prepare dataset
    data_bunch, test_labels, num_query = get_data_bunch(cfg)

    # prepare model
    model = build_model(cfg, data_bunch.c)

    opt_func = partial(torch.optim.Adam)

    def warmup_multistep(start: float, end: float, pct: float) -> float:
        warmup_factor = 1
        gamma = cfg.SOLVER.GAMMA
        milestones = [1.0 * s / cfg.SOLVER.MAX_EPOCHS for s in cfg.SOLVER.STEPS]
        warmup_iter = 1.0 * cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_EPOCHS
        if pct < warmup_iter:
            alpha = pct / warmup_iter
            warmup_factor = cfg.SOLVER.WARMUP_FACTOR * (1 - alpha) + alpha
        return start * warmup_factor * gamma ** bisect_right(milestones, pct)

    lr_sched = Scheduler((cfg.SOLVER.BASE_LR, 0), cfg.SOLVER.MAX_EPOCHS, warmup_multistep)

    loss_func = make_loss(cfg)

    do_train(
        cfg,
        model,
        data_bunch,
        test_labels,
        opt_func,
        lr_sched,
        loss_func,
        num_query
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
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

    sys.stdout = Logger(os.path.join(cfg.OUTPUT_DIR, 'log.txt'))
    print(args)

    if args.config_file != "":
        print("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            print(config_str)
    print("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
