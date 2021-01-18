# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import logging
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.modeling import build_model
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

from fastattr import *


class Trainer(DefaultTrainer):

    def build_model(self, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg, sample_weights=self.sample_weights)
        logger = logging.getLogger("fastreid.attr_model")
        logger.info("Model:\n{}".format(model))
        return model

    def build_train_loader(self, cfg):
        data_loader = build_attr_train_loader(cfg)
        self.sample_weights = data_loader.dataset.sample_weights
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_attr_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, AttrEvaluator(cfg, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
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
