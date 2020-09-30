# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fastreid.engine import DefaultTrainer
from fastreid.utils.file_io import PathManager
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from .config import update_model_teacher_config


class KDTrainer(DefaultTrainer):
    """
    A knowledge distillation trainer for person reid of task.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

        model_t = self.build_model_teacher(self.cfg)
        for param in model_t.parameters():
            param.requires_grad = False

        logger = logging.getLogger('fastreid.' + __name__)

        # Load pre-trained teacher model
        logger.info("Loading teacher model ...")
        Checkpointer(model_t).load(cfg.MODEL.TEACHER_WEIGHTS)

        if PathManager.exists(cfg.MODEL.STUDENT_WEIGHTS):
            logger.info("Loading student model ...")
            Checkpointer(self.model).load(cfg.MODEL.STUDENT_WEIGHTS)
        else:
            logger.info("No student model checkpoints")

        self.model_t = model_t

    def run_step(self):
        """
        Implement the moco training logic described above.
        """
        assert self.model.training, "[KDTrainer] base model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        outs = self.model(data)

        # Compute reid loss
        if isinstance(self.model, DistributedDataParallel):
            loss_dict = self.model.module.losses(outs)
        else:
            loss_dict = self.model.losses(outs)

        with torch.no_grad():
            outs_t = self.model_t(data)

        q_logits = outs["outputs"]["pred_class_logits"]
        t_logits = outs_t["outputs"]["pred_class_logits"].detach()
        loss_dict['loss_kl'] = self.distill_loss(q_logits, t_logits, t=16)

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
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    @classmethod
    def build_model_teacher(cls, cfg) -> nn.Module:
        cfg_t = update_model_teacher_config(cfg)
        model_t = build_model(cfg_t)
        return model_t

    @staticmethod
    def pkt_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
        return loss

    @staticmethod
    def distill_loss(y_s, y_t, t=4):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (t ** 2) / y_s.shape[0]
        return loss
