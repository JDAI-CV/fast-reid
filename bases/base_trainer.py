# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

from utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, tb_writer):
        self.model = model
        self.criterion = criterion
        self.tb_writer = tb_writer

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            inputs, targets = self._parse_data(inputs)
            loss = self._forward(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(loss.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.tb_writer.add_scalar('loss', loss.item(), global_step)
            self.tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        param_group = optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3e}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError
