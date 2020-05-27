# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import logging
import sys

sys.path.append('.')

from torch import nn

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import hooks

from partialreid import *


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DsrEvaluator(cfg, num_query)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    logger = logging.getLogger('fastreid.' + __name__)
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Trainer.build_model(cfg)
        model = nn.DataParallel(model)
        model = model.cuda()

        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(model):
            prebn_cfg = cfg.clone()
            prebn_cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
            prebn_cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN
            logger.info("Prepare precise BN dataset")
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                model,
                # Build a new data loader to not affect training
                Trainer.build_train_loader(prebn_cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ).update_stats()
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
