#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
from functools import partial

import ConfigSpace as CS
import ray
from hyperopt import hp
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch

sys.path.append('.')

from fastreid.config import get_cfg, CfgNode
from fastreid.engine import hooks
from fastreid.modeling import build_model
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.utils.events import CommonMetricPrinter
from fastreid.utils.file_io import PathManager

from hporeid import *

logger = logging.getLogger("fastreid.project.tune")

ray.init(dashboard_host='127.0.0.1')


class HyperTuneTrainer(DefaultTrainer):
    def build_hooks(self):
        r"""
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.MODEL.FREEZE_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            freeze_layers = ",".join(cfg.MODEL.FREEZE_LAYERS)
            logger.info(f'Freeze layer group "{freeze_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iterations')
            ret.append(hooks.FreezeLayer(
                self.model,
                self.optimizer,
                cfg.MODEL.FREEZE_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(TuneReportHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter([CommonMetricPrinter(self.max_iter)], 200))

        return ret

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model


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


def update_config(cfg, config):
    cfg.defrost()

    # lr, weight decay
    # cfg.SOLVER.BASE_LR = config["lr"]
    # cfg.SOLVER.ETA_MIN_LR = config["lr"] * 0.0022
    # cfg.SOLVER.DELAY_ITERS = config["delay_iters"]
    # cfg.SOLVER.WEIGHT_DECAY = config["wd"]
    # cfg.SOLVER.WEIGHT_DECAY_BIAS = config["wd_bias"]

    # batch size, number of instance
    cfg.SOLVER.IMS_PER_BATCH = config["bsz"]
    cfg.DATALOADER.NUM_INSTANCE = config["num_inst"]

    # loss related
    # cfg.MODEL.LOSSES.CE.SCALE = config["ce_scale"]
    # cfg.MODEL.HEADS.SCALE = config["circle_scale"]
    # cfg.MODEL.HEADS.MARGIN = config["circle_margin"]

    # data augmentation
    # cfg.INPUT.DO_AUTOAUG = config["autoaug_enabled"]
    # cfg.INPUT.CJ.ENABLED = config["cj_enabled"]
    return cfg


def train_reid_tune(config, checkpoint_dir=None, cfg=None):
    update_config(cfg, config)

    trainer = HyperTuneTrainer(cfg)
    # Load checkpoint if specific
    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint.pth")
        checkpoint = trainer.checkpointer.resume_or_load(path, resume=False)
        trainer.start_iter = checkpoint.get("iteration", -1) + 1

    # Regular model training
    trainer.train()


def main(args):
    cfg = setup(args)

    exp_metrics = dict(metric="score", mode="max")

    if args.srch_algo == "hyperopt":
        # Create a HyperOpt search space
        search_space = {
            # "lr": hp.loguniform("lr", 1e-6, 1e-3),
            # "delay_iters": hp.randint("delay_iters", 40) + 10,
            # "wd": hp.uniform("wd", 0, 1e-3),
            # "wd_bias": hp.uniform("wd_bias", 0, 1e-3),
            "bsz": hp.choice("bsz", [64, 96, 128, 160, 224, 256]),
            "num_inst": hp.choice("num_inst", [2, 4, 8, 16, 32]),
            # "ce_scale": hp.uniform("ce_scale", 0.1, 1.0),
            # "circle_scale": hp.choice("circle_scale", [16, 32, 64, 128, 256]),
            # "circle_margin": hp.uniform("circle_margin", 0, 1) * 0.4 + 0.1,
            # "autoaug_enabled": hp.choice("autoaug_enabled", [True, False]),
            # "cj_enabled": hp.choice("cj_enabled", [True, False]),
        }

        search_algo = HyperOptSearch(search_space, **exp_metrics)

        if args.pbt:
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                **exp_metrics,
                perturbation_interval=2,
                hyperparam_mutations={
                    "bsz": [64, 96, 128, 160, 224, 256],
                    "num_inst": [2, 4, 8, 16, 32],
                }
            )
        else:
            scheduler = ASHAScheduler(
                metric="score",
                mode="max",
                max_t=10,
                grace_period=1,
                reduction_factor=2)

    elif args.srch_algo == "bohb":
        search_space = CS.ConfigurationSpace()
        search_space.add_hyperparameters([
            # CS.UniformFloatHyperparameter(name="lr", lower=1e-6, upper=1e-2, log=True),
            # CS.UniformIntegerHyperparameter(name="delay_iters", lower=20, upper=60),
            # CS.UniformFloatHyperparameter(name="ce_scale", lower=0.1, upper=1.0),
            # CS.UniformIntegerHyperparameter(name="circle_scale", lower=8, upper=256),
            # CS.UniformFloatHyperparameter(name="circle_margin", lower=0.1, upper=0.5),
            # CS.UniformFloatHyperparameter(name="wd", lower=0, upper=1e-3),
            # CS.UniformFloatHyperparameter(name="wd_bias", lower=0, upper=1e-3),
            CS.CategoricalHyperparameter(name="bsz", choices=[64, 96, 128, 160, 224, 256]),
            CS.CategoricalHyperparameter(name="num_inst", choices=[2, 4, 8, 16, 32]),
            # CS.CategoricalHyperparameter(name="autoaug_enabled", choices=[True, False]),
            # CS.CategoricalHyperparameter(name="cj_enabled", choices=[True, False]),
        ])

        search_algo = TuneBOHB(
            search_space, max_concurrent=4, **exp_metrics)

        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            reduction_factor=3,
            max_t=9,
            **exp_metrics,
        )

    else:
        raise ValueError("Search algorithm must be chosen from [hyperopt, bohb], but got {}".format(args.srch_algo))

    reporter = CLIReporter(
        parameter_columns=["bsz", "num_inst"],
        metric_columns=["r1", "map", "training_iteration"])

    analysis = tune.run(
        partial(
            train_reid_tune,
            cfg=cfg),
        resources_per_trial={"cpu": 12, "gpu": 1},
        search_alg=search_algo,
        num_samples=args.num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=cfg.OUTPUT_DIR,
        keep_checkpoints_num=10,
        name=args.srch_algo)

    best_trial = analysis.get_best_trial("score", "max", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation mAP: {}, Rank-1: {}".format(
        best_trial.last_result["map"], best_trial.last_result["r1"]))

    save_dict = dict(R1=best_trial.last_result["r1"].item(), mAP=best_trial.last_result["map"].item())
    save_dict.update(best_trial.config)
    path = os.path.join(cfg.OUTPUT_DIR, "best_config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(CfgNode(save_dict).dump())
    logger.info("Best config saved to {}".format(os.path.abspath(path)))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--num-trials", type=int, default=12, help="number of tune trials")
    parser.add_argument("--srch-algo", type=str, default="bohb",
                        help="search algorithms for hyperparameters search space")
    parser.add_argument("--pbt", action="store_true", help="use population based training")
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
