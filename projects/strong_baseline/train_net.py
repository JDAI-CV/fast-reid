# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')
from fastreid.config import cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.evaluation import ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        # if output_folder is None:
        #     output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

    # log_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST_NAMES, cfg.MODEL.VERSION)
    # if not os.path.exists(log_save_dir):
    #     os.makedirs(log_save_dir)
    #
    # logger = setup_logger(cfg.MODEL.VERSION, log_save_dir, 0)
    # logger.info("Using {} GPUs.".format(num_gpus))
    # logger.info(args)
    #
    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    # logger.info("Running with config:\n{}".format(cfg))
    #
    # logger.info('start training')
    # cudnn.benchmark = True
