# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY, build_model, Baseline
from fastreid.utils.checkpoint import Checkpointer

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class Distiller(Baseline):
    def __init__(self, cfg):
        super(Distiller, self).__init__(cfg)

        # Get teacher model config
        cfg_t = get_cfg()
        cfg_t.merge_from_file(cfg.KD.MODEL_CONFIG)

        model_t = build_model(cfg_t)
        logger.info("Teacher model:\n{}".format(model_t))

        # No gradients for teacher model
        for param in model_t.parameters():
            param.requires_grad_(False)

        logger.info("Loading teacher model weights ...")
        Checkpointer(model_t).load(cfg.KD.MODEL_WEIGHTS)

        # Not register teacher model as `nn.Module`, this is
        # make sure teacher model weights not saved
        self.model_t = [model_t.backbone, model_t.heads]

    def forward(self, batched_inputs):
        if self.training:
            images = self.preprocess_image(batched_inputs)
            # student model forward
            s_feat = self.backbone(images)
            assert "targets" in batched_inputs, "Labels are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            if targets.sum() < 0: targets.zero_()

            s_outputs = self.heads(s_feat, targets)

            # teacher model forward
            with torch.no_grad():
                t_feat = self.model_t[0](images)
                t_outputs = self.model_t[1](t_feat, targets)

            losses = self.losses(s_outputs, t_outputs, targets)
            return losses

        # Eval mode, just conventional reid feature extraction
        else:
            return super(Distiller, self).forward(batched_inputs)

    def losses(self, s_outputs, t_outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super(Distiller, self).losses(s_outputs, gt_labels)

        s_logits = s_outputs["pred_class_logits"]
        t_logits = t_outputs["pred_class_logits"].detach()
        loss_dict["loss_jsdiv"] = self.jsdiv_loss(s_logits, t_logits)

        return loss_dict

    @staticmethod
    def _kldiv(y_s, y_t, t):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (t ** 2) / y_s.shape[0]
        return loss

    def jsdiv_loss(self, y_s, y_t, t=16):
        loss = (self._kldiv(y_s, y_t, t) + self._kldiv(y_t, y_s, t)) / 2
        return loss
