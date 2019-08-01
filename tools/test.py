# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import get_data_bunch
from engine.inference import inference
from utils.logger import setup_logger
from modeling import build_model


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
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR): os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    data_bunch, test_labels, num_query = get_data_bunch(cfg)
    # model = build_model(cfg, data_bunch.c)
    # state_dict = torch.load(cfg.TEST.WEIGHT)
    # model.load_state_dict(state_dict['model'])
    # model.cuda()
    model = torch.jit.load("/export/home/lxy/reid_baseline/pcb_model_v0.2.pt")

    inference(cfg, model, data_bunch, test_labels, num_query)


if __name__ == '__main__':
    main()
