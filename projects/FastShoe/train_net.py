# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 10:12:28
# @Author  : zuchen.wang@vipshop.com
# @File    : train_net.py.py

import json
import logging
import os
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer, PathManager
from fastreid.utils import bughook

from fastshoe import PairTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = PairTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        try:
            output_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
            path = os.path.join(output_dir, "idx2class.json")
            with PathManager.open(path, 'r') as f:
                idx2class = json.load(f)
        except:
            logger = logging.getLogger(__name__)
            logger.info(f"Cannot find idx2class dict in {os.path.dirname(cfg.MODEL.WEIGHTS)}")

        res = PairTrainer.test(cfg, model)
        return res

    trainer = PairTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
