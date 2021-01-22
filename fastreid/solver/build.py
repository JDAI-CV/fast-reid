# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

from . import lr_scheduler
from . import optim


def build_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad: continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "heads" in key:
            lr *= cfg.SOLVER.HEADS_LR_FACTOR
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay}]

    solver_opt = cfg.SOLVER.OPT
    if solver_opt == "SGD":
        opt_fns = getattr(optim, solver_opt)(
            params,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=True if cfg.SOLVER.MOMENTUM and cfg.SOLVER.NESTEROV else False
        )
    else:
        opt_fns = getattr(optim, solver_opt)(params)
    return opt_fns


def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    max_epoch = cfg.SOLVER.MAX_EPOCH - max(
        math.ceil(cfg.SOLVER.WARMUP_ITERS / iters_per_epoch), cfg.SOLVER.DELAY_EPOCHS)

    scheduler_dict = {}

    scheduler_args = {
        "MultiStepLR": {
            "optimizer": optimizer,
            # multi-step lr scheduler options
            "milestones": cfg.SOLVER.STEPS,
            "gamma": cfg.SOLVER.GAMMA,
        },
        "CosineAnnealingLR": {
            "optimizer": optimizer,
            # cosine annealing lr scheduler options
            "T_max": max_epoch,
            "eta_min": cfg.SOLVER.ETA_MIN_LR,
        },

    }

    scheduler_dict["lr_sched"] = getattr(lr_scheduler, cfg.SOLVER.SCHED)(
        **scheduler_args[cfg.SOLVER.SCHED])

    if cfg.SOLVER.WARMUP_ITERS > 0:
        warmup_args = {
            "optimizer": optimizer,

            # warmup options
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "warmup_method": cfg.SOLVER.WARMUP_METHOD,
        }
        scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict
