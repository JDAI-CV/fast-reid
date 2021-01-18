# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch import nn

from fastreid.modeling.meta_arch import META_ARCH_REGISTRY, Distiller

logger = logging.getLogger("fastreid.meta_arch.overhaul_distiller")


def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = F.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / \
                          math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.tensor(margin, dtype=torch.float32, device=mean.device)


@META_ARCH_REGISTRY.register()
class DistillerOverhaul(Distiller):
    def __init__(self, cfg):
        super().__init__(cfg)

        s_channels = self.backbone.get_channel_nums()
        t_channels = self.model_t[0].get_channel_nums()

        self.connectors = nn.ModuleList(
            [build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = self.model_t[0].get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer("margin%d" % (i + 1),
                                 margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

    def forward(self, batched_inputs):
        if self.training:
            images = self.preprocess_image(batched_inputs)
            # student model forward
            s_feats, s_feat = self.backbone.extract_feature(images, preReLU=True)
            assert "targets" in batched_inputs, "Labels are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            if targets.sum() < 0: targets.zero_()

            s_outputs = self.heads(s_feat, targets)

            # teacher model forward
            with torch.no_grad():
                t_feats, t_feat = self.model_t[0].extract_feature(images, preReLU=True)
                t_outputs = self.model_t[1](t_feat, targets)

            losses = self.losses(s_outputs, s_feats, t_outputs, t_feats, targets)
            return losses

        else:
            outputs = super(DistillerOverhaul, self).forward(batched_inputs)
            return outputs

    def losses(self, s_outputs, s_feats, t_outputs, t_feats, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(s_outputs, t_outputs, gt_labels)

        # Overhaul distillation loss
        feat_num = len(s_feats)
        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(
                self, "margin%d" % (i + 1)).to(s_feats[i].dtype)) / 2 ** (feat_num - i - 1)

        loss_dict["loss_overhaul"] = loss_distill / len(gt_labels) / 10000

        return loss_dict
