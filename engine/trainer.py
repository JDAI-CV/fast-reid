# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os

import matplotlib.pyplot as plt
from fastai.vision import *
from .callbacks import *


def do_train(
        cfg,
        model,
        data_bunch,
        test_labels,
        opt_func,
        lr_sched,
        loss_func,
        num_query,
):
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = Path(cfg.OUTPUT_DIR)
    epochs = cfg.SOLVER.MAX_EPOCHS
    total_iter = len(data_bunch.train_dl)

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start Training")

    cb_fns = [
        partial(LRScheduler, lr_sched=lr_sched),
        partial(TestModel, test_labels=test_labels, eval_period=eval_period, num_query=num_query, logger=logger),
    ]

    learn = Learner(
        data_bunch,
        model,
        path=output_dir,
        opt_func=opt_func,
        loss_func=loss_func,
        true_wd=False,
        callback_fns=cb_fns,
        callbacks=[TrackValue(logger, total_iter)])

    # continue training
    if cfg.MODEL.CHECKPOINT is not '':
        state = torch.load(cfg.MODEL.CHECKPOINT)
        if set(state.keys()) == {'model', 'opt'}:
            model_state = state['model']
            learn.model.load_state_dict(model_state)
            learn.create_opt(0, 0)
            learn.opt.load_state_dict(state['opt'])
        else:
            learn.model.load_state_dict(state['model'])
        logger.info(f'continue training from checkpoint {cfg.MODEL.CHECKPOINT}')

    learn.fit(epochs, lr=cfg.SOLVER.BASE_LR, wd=cfg.SOLVER.WEIGHT_DECAY)
