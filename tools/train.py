# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys
from pprint import pprint

import torch
from torch import nn

import network
from core.config import opt, update_config
from core.loader import get_data_provider
from core.solver import Solver
from utils.lr_scheduler import LRScheduler

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def train(args):
    logging.info('======= user config ======')
    logging.info(pprint(opt))
    logging.info(pprint(args))
    logging.info('======= end ======')

    train_data, test_data, num_query = get_data_provider(opt)

    net = getattr(network, opt.network.name)(opt.dataset.num_classes, opt.network.last_stride)
    net = nn.DataParallel(net).cuda()

    optimizer = getattr(torch.optim, opt.train.optimizer)(net.parameters(), lr=opt.train.lr, weight_decay=opt.train.wd)
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = None

    def loss_fn(scores, feat, labels):
        ce = ce_loss(scores, labels)
        # triplet = triplet_loss(feat, labels)
        # return ce + triplet
        return ce

    lr_scheduler = LRScheduler(base_lr=opt.train.lr, step=opt.train.step,
                               factor=opt.train.factor, warmup_epoch=opt.train.warmup_epoch,
                               warmup_begin_lr=opt.train.warmup_begin_lr)

    mod = Solver(opt, net)
    mod.fit(train_data=train_data, test_data=test_data, num_query=num_query, optimizer=optimizer,
            criterion=loss_fn, lr_scheduler=lr_scheduler)


def main():
    parser = argparse.ArgumentParser(description='reid model training')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('--save_dir', type=str, default=None, required=True,
                        help='model save checkpoint directory')

    args = parser.parse_args()
    if args.config_file is not None:
        update_config(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.network.gpus
    train(args)


if __name__ == '__main__':
    main()
