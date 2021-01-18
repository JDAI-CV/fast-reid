# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
from collections import OrderedDict

import torch

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example if you want to"
                      "train with DDP")

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

logger = logging.getLogger("fastreid")


def get_evaluator(cfg, dataset_name, output_dir=None):
    data_loader, num_query = build_reid_test_loader(cfg, dataset_name)
    return data_loader, ReidEvaluator(cfg, num_query, output_dir)


def do_test(cfg, model):
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        logger.info("Prepare testing set")
        try:
            data_loader, evaluator = get_evaluator(cfg, dataset_name)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. implement its `build_evaluator` method."
            )
            results[dataset_name] = {}
            continue
        results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP_ENABLED)
        results[dataset_name] = results_i

    if comm.is_main_process():
        assert isinstance(
            results, dict
        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
            results
        )
        print_csv_format(results)

    if len(results) == 1: results = list(results.values())[0]

    return results


def do_train(cfg, model, resume=False):
    data_loader = build_reid_train_loader(cfg)

    model.train()
    optimizer = build_optimizer(cfg, model)

    optimizer_ckpt = dict(optimizer=optimizer)

    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(
        model,
        cfg.OUTPUT_DIR,
        save_to_disk=comm.is_main_process(),
        **optimizer_ckpt,
        **scheduler
    )

    iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH

    start_epoch = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("epoch", -1) + 1
    )
    iteration = start_iter = start_epoch * iters_per_epoch

    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_epoch)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR)
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support some hooks, such as
    # accurate timing, FP16 training and precise BN here,
    # because they are not trivial to implement in a small training loop
    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            storage.epoch = epoch
            for data, _ in zip(data_loader, range(iters_per_epoch)):
                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if iteration - start_iter > 5 and (
                        (iteration + 1) % 200 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()

                iteration += 1

            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if (epoch + 1) <= warmup_epochs:
                scheduler["warmup_sched"].step()
            elif (epoch + 1) >= delay_epochs:
                scheduler["lr_sched"].step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and epoch != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage

            periodic_checkpointer.step(epoch)


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

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


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
