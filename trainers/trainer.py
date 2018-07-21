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


class clsTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, _ = self.model(self.data)
        self.loss = self.criterion(score, self.target)

    def _backward(self):
        self.loss.backward()


class tripletTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        feat = self.model(self.data)
        self.loss = self.criterion(feat, self.target)

    def _backward(self):
        self.loss.backward()


class cls_tripletTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat = self.model(self.data)
        self.loss = self.criterion(score, feat, self.target)

    def _backward(self):
        self.loss.backward()
