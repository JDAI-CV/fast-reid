# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import get_test_dataloader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
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
    # set pretrian = False to avoid loading weight repeatedly
    cfg.MODEL.PRETRAIN = False
    cfg.freeze()

    logger = setup_logger("reid_baseline", False, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    model = build_model(cfg, 0)
    model = model.cuda()
    model.load_params_wo_fc(torch.load(cfg.TEST.WEIGHT))

    test_dataloader, num_query = get_test_dataloader(cfg)

    inference(cfg, model, test_dataloader, num_query)


if __name__ == '__main__':
    main()

