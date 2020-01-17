# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import shutil
import logging
import weakref
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import get_dataloader
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher, test_data_prefetcher
from modeling import build_model
from solver.build import make_lr_scheduler, make_optimizer
from utils.meters import AverageMeter
from torch.optim.lr_scheduler import CyclicLR
from apex import amp


# class HookBase:
#     """
#     Base class for hooks that can be registered with :class:`TrainerBase`.
#     Each hook can implement 4 methods. The way they are called is demonstrated
#     in the following snippet:
#     .. code-block:: python
#         hook.before_train()
#         for iter in range(start_iter, max_iter):
#             hook.before_step()
#             trainer.run_step()
#             hook.after_step()
#         hook.after_train()
#     Notes:
#         1. In the hook method, users can access `self.trainer` to access more
#            properties about the context (e.g., current iteration).
#         2. A hook that does something in :meth:`before_step` can often be
#            implemented equivalently in :meth:`after_step`.
#            If the hook takes non-trivial time, it is strongly recommended to
#            implement the hook in :meth:`after_step` instead of :meth:`before_step`.
#            The convention is that :meth:`before_step` should only take negligible time.
#            Following this convention will allow hooks that do care about the difference
#            between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
#            function properly.
#     Attributes:
#         trainer: A weak reference to the trainer object. Set by the trainer when the hook is
#             registered.
#     """
#
#     def before_train(self):
#         """
#         Called before the first iteration.
#         """
#         pass
#
#     def after_train(self):
#         """
#         Called after the last iteration.
#         """
#         pass
#
#     def before_step(self):
#         """
#         Called before each iteration.
#         """
#         pass
#
#     def after_step(self):
#         """
#         Called after each iteration.
#         """
#         pass


# class TrainerBase:
#     """
#     Base class for iterative trainer with hooks.
#     The only assumption we made here is: the training runs in a loop.
#     A subclass can implement what the loop is.
#     We made no assumptions about the existence of dataloader, optimizer, model, etc.
#     Attributes:
#         iter(int): the current iteration.
#         start_iter(int): The iteration to start with.
#             By convention the minimum possible value is 0.
#         max_iter(int): The iteration to end training.
#         storage(EventStorage): An EventStorage that's opened during the course of training.
#     """
#
#     def __init__(self):
#         self._hooks = []
#
#     def register_hooks(self, hooks):
#         """
#         Register hooks to the trainer. The hooks are executed in the order
#         they are registered.
#         Args:
#             hooks (list[Optional[HookBase]]): list of hooks
#         """
#         hooks = [h for h in hooks if h is not None]
#         for h in hooks:
#             assert isinstance(h, HookBase)
#             # To avoid circular reference, hooks and trainer cannot own each other.
#             # This normally does not matter, but will cause memory leak if the
#             # involved objects contain __del__:
#             # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
#             h.trainer = weakref.proxy(self)
#         self._hooks.extend(hooks)
#
#     def train(self, start_iter: int, max_iter: int):
#         """
#         Args:
#             start_iter, max_iter (int): See docs above
#         """
#         logger = logging.getLogger(__name__)
#         logger.info("Starting training from iteration {}".format(start_iter))
#
#         self.iter = self.start_iter = start_iter
#         self.max_iter = max_iter
#
#         with EventStorage(start_iter) as self.storage:
#             try:
#                 self.before_train()
#                 for self.iter in range(start_iter, max_iter):
#                     self.before_step()
#                     self.run_step()
#                     self.after_step()
#             finally:
#                 self.after_train()
#
#     def before_train(self):
#         for h in self._hooks:
#             h.before_train()
#
#     def after_train(self):
#         for h in self._hooks:
#             h.after_train()
#
#     def before_step(self):
#         for h in self._hooks:
#             h.before_step()
#
#     def after_step(self):
#         for h in self._hooks:
#             h.after_step()
#         # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
#         self.storage.step()
#
#     def run_step(self):
#         raise NotImplementedError


class ReidSystem():
    def __init__(self, cfg, logger, writer):
        self.cfg, self.logger, self.writer = cfg, logger, writer
        # Define dataloader
        self.tng_dataloader, self.val_dataloader, self.num_classes, self.num_query = get_dataloader(cfg)
        # networks
        self.model = build_model(cfg, self.num_classes)

        self._construct()

    def _construct(self):
        self.global_step = 0
        self.current_epoch = 0
        self.batch_nb = 0
        self.max_epochs = self.cfg.SOLVER.MAX_EPOCHS
        self.log_interval = self.cfg.SOLVER.LOG_INTERVAL
        self.eval_period = self.cfg.SOLVER.EVAL_PERIOD
        self.use_dp = False
        self.use_ddp = False
        self.use_mask = self.cfg.INPUT.USE_MASK

    def on_train_begin(self):
        self.best_mAP = -np.inf
        self.running_loss = AverageMeter()
        log_save_dir = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.DATASETS.TEST_NAMES, self.cfg.MODEL.VERSION)
        self.model_save_dir = os.path.join(log_save_dir, 'ckpts')
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)

        self.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        self.use_dp = (len(self.gpus) > 1) and (self.cfg.MODEL.DIST_BACKEND == 'dp')

        self.model.cuda()
        # optimizer and scheduler
        self.opt = make_optimizer(self.cfg, self.model)

        if self.use_dp:
            self.model = nn.DataParallel(self.model)

        # self.model, self.opt = amp.initialize(self.model, self.opt, opt_level='O1')
        self.lr_sched = make_lr_scheduler(self.cfg, self.opt)

        self.model.train()

    def on_epoch_begin(self):
        self.batch_nb = 0
        self.current_epoch += 1
        self.t0 = time.time()
        self.running_loss.reset()

        self.tng_prefetcher = data_prefetcher(self.tng_dataloader)

        # if self.current_epoch == 1:
        #     # freeze for first 10 epochs
        #     if self.use_dp or self.use_ddp:
        #         self.model.module.unfreeze_specific_layer(['bottleneck', 'classifier'])
        #     else:
        #         self.model.unfreeze_specific_layer(['bottleneck', 'classifier'])
        # elif self.current_epoch == 11:
        #     if self.use_dp or self.use_ddp:
        #         self.model.module.unfreeze_all_layers()
        #     else:
        #         self.model.unfreeze_all_layers()

    def training_step(self, batch):
        if self.use_mask:
            inputs, masks, labels, _ = batch
        else:
            inputs, labels, _ = batch
            masks = None
        outputs = self.model(inputs, labels, pose=masks)
        if self.use_dp or self.use_ddp:
            loss_dict = self.model.module.getLoss(outputs, labels, mask_labels=masks)
            total_loss = self.model.module.loss
        else:
            loss_dict = self.model.getLoss(outputs, labels, mask_labels=masks)
            total_loss = self.model.loss

        print_str = f'\r Epoch {self.current_epoch} Iter {self.batch_nb}/{len(self.tng_prefetcher.loader)} '
        for loss_name, loss_value in loss_dict.items():
            print_str += (loss_name + f': {loss_value.item():.3f} ')
        print_str += f'Total loss: {total_loss.item():.3f} '
        print(print_str, end=' ')

        if self.writer is not None:
            if (self.global_step + 1) % self.log_interval == 0:
                for loss_name, loss_value in loss_dict.items():
                    self.writer.add_scalar(loss_name, loss_value.item(), self.global_step)

        self.running_loss.update(total_loss.item())

        self.opt.zero_grad()
        # with amp.scale_loss(total_loss, self.opt) as scaled_loss:
        #     scaled_loss.backward()
        total_loss.backward()
        self.opt.step()
        self.global_step += 1
        self.batch_nb += 1

    def on_epoch_end(self):
        elapsed = time.time() - self.t0
        mins = int(elapsed) // 60
        seconds = int(elapsed - mins * 60)
        print('')
        self.logger.info(f'Epoch {self.current_epoch} Total loss: {self.running_loss.avg:.3f} '
                         f'lr: {self.opt.param_groups[0]["lr"]:.2e} During {mins:d}min:{seconds:d}s')
        # update learning rate
        self.lr_sched.step()

    @torch.no_grad()
    def test(self):
        # convert to eval mode
        self.model.eval()

        feats, pids, camids = [], [], []
        val_prefetcher = data_prefetcher(self.val_dataloader)
        batch = val_prefetcher.next()
        while batch[0] is not None:
            # if self.use_mask:
            #     inputs, masks, pid, camid = batch
            # else:
            inputs, pid, camid = batch
                # masks = None
            # img, pid, camid = batch
            feat = self.model(inputs, pose=None)
            feats.append(feat)
            pids.extend(np.asarray(pid.cpu().numpy()))
            camids.extend(np.asarray(camid))

            batch = val_prefetcher.next()

        feats = torch.cat(feats, dim=0)
        if self.cfg.TEST.NORM:
            feats = F.normalize(feats, p=2, dim=1)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(pids[self.num_query:])
        g_camids = np.asarray(camids[self.num_query:])

        # m, n = qf.shape[0], gf.shape[0]
        distmat = torch.mm(qf, gf.t()).cpu().numpy()
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.numpy()
        cmc, mAP = evaluate(1-distmat, q_pids, g_pids, q_camids, g_camids)
        self.logger.info(f"Test Results - Epoch: {self.current_epoch}")
        self.logger.info(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            self.logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

        if self.writer is not None:
            self.writer.add_scalar('rank1', cmc[0], self.global_step)
            self.writer.add_scalar('mAP', mAP, self.global_step)
        metric_dict = {'rank1': cmc[0], 'mAP': mAP}
        # convert to train mode
        self.model.train()
        return metric_dict

    def train(self):
        self.on_train_begin()
        for epoch in range(self.max_epochs):
            self.on_epoch_begin()
            batch = self.tng_prefetcher.next()
            while batch[0] is not None:
                self.training_step(batch)
                batch = self.tng_prefetcher.next()
            self.on_epoch_end()
            if self.eval_period > 0 and ((epoch + 1) % self.eval_period == 0):
                metric_dict = self.test()
                if metric_dict['mAP'] > self.best_mAP:
                    is_best = True
                    self.best_mAP = metric_dict['mAP']
                else:
                    is_best = False
                self.save_checkpoints(is_best)

            torch.cuda.empty_cache()

    def save_checkpoints(self, is_best):
        if self.use_dp:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        # TODO: add optimizer state dict and lr scheduler
        filepath = os.path.join(self.model_save_dir, f'model_epoch{self.current_epoch}.pth')
        torch.save(state_dict, filepath)
        if is_best:
            best_filepath = os.path.join(self.model_save_dir, 'model_best.pth')
            shutil.copyfile(filepath, best_filepath)
