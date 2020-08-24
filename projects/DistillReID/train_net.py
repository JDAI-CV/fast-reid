#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock, guan'an wang
@contact: sherlockliao01@gmail.com, guan.wang0706@gmail.com
"""

import sys
import torch
from torch import nn

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, DefaultTrainer, launch
from fastreid.utils.checkpoint import Checkpointer

from kdreid import *


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_shufflenet_config(cfg)
    add_kdreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
        res = DefaultTrainer.test(cfg, model)
        return res

    if args.kd: trainer = KDTrainer(cfg)
    else:       trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--kd", action="store_true", help="kd training with teacher model guided")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )