# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import time
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
from fastreid.engine import DefaultTrainer
from .data_build import build_attr_train_loader, build_attr_test_loader
from .attr_evaluation import AttrEvaluator


class AttrTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Sample weight for attributed imbalanced classification
        bce_weight_enabled = self.cfg.MODEL.LOSSES.BCE.WEIGHT_ENABLED
        # fmt: off
        if bce_weight_enabled: self.sample_weights = self.data_loader.dataset.sample_weights.to("cuda")
        else:                  self.sample_weights = None
        # fmt: on

    @classmethod
    def build_train_loader(cls, cfg):
        return build_attr_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_attr_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, AttrEvaluator(cfg, output_folder)

    def run_step(self):
        r"""
        Implement the attribute model training logic.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """

        with amp.autocast(enabled=self.amp_enabled):
            outs = self.model(data)

            # Compute loss
            if isinstance(self.model, DistributedDataParallel):
                loss_dict = self.model.module.losses(outs, self.sample_weights)
            else:
                loss_dict = self.model.losses(outs, self.sample_weights)

            losses = sum(loss_dict.values())

        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp_enabled:
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses.backward()
            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method.
            """
            self.optimizer.step()
