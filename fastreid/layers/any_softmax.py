# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Linear',
    'ArcSoftmax',
    'CosSoftmax',
    'CircleSoftmax'
]


class Linear(nn.Module):
    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self._num_classes = num_classes
        self.s = 1
        self.m = 0

    def forward(self, logits, *args):
        return logits

    def extra_repr(self):
        return 'num_classes={}, scale={}, margin={}'.format(self._num_classes, self.s, self.m)


class ArcSoftmax(nn.Module):
    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self._num_classes = num_classes
        self.s = scale
        self.m = margin

        self.easy_margin = False

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, logits, targets):
        sine = torch.sqrt(1.0 - torch.pow(logits, 2))
        phi = logits * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(logits > 0, phi, logits)
        else:
            phi = torch.where(logits > self.threshold, phi, logits - self.mm)
        one_hot = torch.zeros(logits.size(), device=logits.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * logits)
        output *= self.s
        return output

    def extra_repr(self):
        return 'num_classes={}, scale={}, margin={}'.format(self._num_classes, self.s, self.m)


class CircleSoftmax(nn.Module):
    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self._num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits, targets):
        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (logits - delta_p)
        s_n = self.s * alpha_n * (logits - delta_n)

        targets = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits

    def extra_repr(self):
        return "num_classes={}, scale={}, margin={}".format(self._num_classes, self.s, self.m)


class CosSoftmax(nn.Module):
    r"""Implement of large margin cosine distance:
    Args:
        num_classes: size of each output sample
    """

    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self._num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits, targets):
        phi = logits - self.m
        targets = F.one_hot(targets, num_classes=self._num_classes)
        output = (targets * phi) + ((1.0 - targets) * logits)
        output *= self.s

        return output

    def extra_repr(self):
        return "num_classes={}, scale={}, margin={}".format(self._num_classes, self.s, self.m)
