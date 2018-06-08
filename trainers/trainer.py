# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from bases.base_trainer import BaseTrainer


class ResNetClsTrainer(BaseTrainer):
    def __init__(self, model, criterion, tb_writer):
        super().__init__(model, criterion, tb_writer)

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        return imgs.cuda(), pids.cuda()

    def _forward(self, inputs, targets):
        cls_score, _ = self.model(inputs)
        loss = self.criterion(cls_score, targets)
        return loss


class ResNetTriTrainer(BaseTrainer):
    def __init__(self, model, criterion, tb_writer):
        super().__init__(model, criterion, tb_writer)

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        return imgs.cuda(), pids.cuda()

    def _forward(self, inputs, targets):
        feat = self.model(inputs)
        loss = self.criterion(feat, targets)
        return loss


class ResNetClsTriTrainer(BaseTrainer):
    def __init__(self, model, criterion, tb_writer):
        super().__init__(model, criterion, tb_writer)

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        return imgs.cuda(), pids.cuda()

    def _forward(self, inputs, targets):
        cls_score, feat = self.model(inputs)
        loss = self.criterion(cls_score, feat, targets)
        return loss