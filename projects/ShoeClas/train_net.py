# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 10:12:28
# @Author  : zuchen.wang@vipshop.com
# @File    : train_net.py.py

import json
import os
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils import bughook
from shoeclas import PairTrainer



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    setattr(cfg, 'eval_only', args.eval_only)
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
        res = PairTrainer.test(cfg, model)
        return res
    else:
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
        dist_url='auto',
        args=(args,),
    )
