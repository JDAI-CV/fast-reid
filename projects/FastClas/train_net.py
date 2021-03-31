#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import sys
import os

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.data.build import build_reid_train_loader, build_reid_test_loader
from fastreid.evaluation.clas_evaluator import ClasEvaluator
from fastreid.utils.checkpoint import Checkpointer, PathManager
from fastreid.utils import comm
from fastreid.engine import DefaultTrainer

from fastclas import *


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger("fastreid.clas_dataset")
        logger.info("Prepare training set")
        data_loader = build_reid_train_loader(cfg, Dataset=ClasDataset)

        # Save index to class dictionary
        output_dir = cfg.OUTPUT_DIR
        if comm.is_main_process() and output_dir:
            path = os.path.join(output_dir, "idx2class.json")
            with PathManager.open(path, "w") as f:
                json.dump(data_loader.dataset.idx_to_class, f)

        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_test_loader(cfg, dataset_name, Dataset=ClasDataset)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, _ = cls.build_test_loader(cfg, dataset_name)
        return data_loader, ClasEvaluator(cfg, output_dir)


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
        model = Trainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

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
