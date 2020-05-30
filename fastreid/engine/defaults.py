# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn import DataParallel

from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation import (DatasetEvaluator, ReidEvaluator,
                                 inference_on_dataset, print_csv_format)
from fastreid.modeling.meta_arch import build_model
from fastreid.layers.sync_bn import patch_replication_callback
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.utils import comm
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from fastreid.utils.file_io import PathManager
from fastreid.utils.logger import setup_logger
from . import hooks
from .train_loop import SimpleTrainer

__all__ = ["default_argument_parser", "default_setup", "DefaultPredictor", "DefaultTrainer"]


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    # parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    # parser.add_argument("--num-machines", type=int, default=1)
    # parser.add_argument(
    #     "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    # )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    # parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    # logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.device = device
        self.model = build_model(self.cfg)
        self.model.to(device)
        self.model.eval()

        checkpointer = Checkpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            image = image.to(self.device)
            inputs = {"images": image}
            predictions = self.model(inputs)
            # Normalize feature to compute cosine distance
            pred_feat = F.normalize(predictions)
            pred_feat = pred_feat.cpu().data
            return pred_feat


class DefaultTrainer(SimpleTrainer):
    """
    A trainer with default training logic. Compared to `SimpleTrainer`, it
    contains the following logic in addition:
    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.
    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.
    The code of this class has been annotated about restrictive assumptions it mades.
    When they do not work for you, you're encouraged to:
    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.
    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.
    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    Examples:
    .. code-block:: python
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self.cfg = cfg
        logger = logging.getLogger(__name__)
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        logger.info('Prepare training set')
        data_loader = self.build_train_loader(cfg)
        # For training, wrap with DP. But don't need this for inference.
        model = DataParallel(model)
        if cfg.MODEL.BACKBONE.NORM == "syncBN":
            # Monkey-patching with syncBN
            patch_replication_callback(model)
        model = model.cuda()
        super().__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            self.data_loader.dataset,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_ITER + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.
        Otherwise, load a model specified by the config.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

        # Reinitialize dataloader iter because when we update dataset person identity dict
        # to resume training, DataLoader won't update this dictionary when using multiprocess
        # because of the function scope.
        self._data_loader_iter = iter(self.data_loader)

        self.start_iter = checkpoint.get("iteration", -1) if resume else -1
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter += 1

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_FACTOR,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if cfg.MODEL.OPEN_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            open_layers = ",".join(cfg.MODEL.OPEN_LAYERS)
            logger.info(f'Open "{open_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iters')
            ret.append(hooks.FreezeLayer(
                self.model,
                cfg.MODEL.OPEN_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_PERIOD))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        .. code-block:: python
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        # if hasattr(self, "_last_eval_results") and comm.is_main_process():
        #     verify_results(self.cfg, self._last_eval_results)
        #     return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, num_query, output_dir=None):
        return ReidEvaluator(cfg, num_query, output_dir)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info(f'prepare test set')
            data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, num_query)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
