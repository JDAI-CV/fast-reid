#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.data.build import build_reid_train_loader, build_reid_test_loader
from fastreid.evaluation.clas_evaluator import ClasEvaluator
from fastreid.utils.checkpoint import Checkpointer, PathManager
from fastreid.utils import comm
from fastreid.engine import DefaultTrainer
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.data.build import _root

from fastclas import *


class ClasTrainer(DefaultTrainer):

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

        train_items = list()
        for d in cfg.DATASETS.NAMES:
            data = DATASET_REGISTRY.get(d)(root=_root)
            if comm.is_main_process():
                data.show_train()
            train_items.extend(data.train)

        transforms = build_transforms(cfg, is_train=True)
        train_set = ClasDataset(train_items, transforms)

        data_loader = build_reid_train_loader(cfg, train_set=train_set)

        # Save index to class dictionary
        output_dir = cfg.OUTPUT_DIR
        if comm.is_main_process() and output_dir:
            path = os.path.join(output_dir, "idx2class.json")
            with PathManager.open(path, "w") as f:
                json.dump(train_set.idx_to_class, f)

        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_test_loader`.
        Overwrite it if you'd like a different data loader.
        """

        data = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            data.show_test()
        transforms = build_transforms(cfg, is_train=False)
        test_set = ClasDataset(data.query, transforms)
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
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
        model = ClasTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = ClasTrainer.test(cfg, model)
        return res

    trainer = ClasTrainer(cfg)

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
