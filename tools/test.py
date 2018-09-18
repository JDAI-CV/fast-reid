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

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def test(args):
    logging.info('======= user config ======')
    logging.info(pprint(opt))
    logging.info(pprint(args))
    logging.info('======= end ======')

    train_data, test_data, num_query = get_data_provider(opt)

    net = getattr(network, opt.network.name)(opt.dataset.num_classes, opt.network.last_stride)
    net = net.load_state_dict(torch.load(args.load_model))
    net = nn.DataParallel(net).cuda()

    mod = Solver(opt, net)
    mod.test_func(test_data, num_query)


def main():
    parser = argparse.ArgumentParser(description='reid model testing')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('--load_model', type=str, required=True,
                        help='load trained model for testing')

    args = parser.parse_args()
    if args.config_file is not None:
        update_config(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.network.gpus
    test(args)


if __name__ == '__main__':
    main()
