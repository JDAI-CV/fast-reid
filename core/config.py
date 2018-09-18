# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import yaml
from easydict import EasyDict as edict

__C = edict()
opt = __C
__C.seed = 0

__C.dataset = edict()
__C.dataset.name = 'market1501'
__C.dataset.num_classes = 751

__C.aug = edict()
__C.aug.resize_size = [256, 128]
__C.aug.color_jitter = False
__C.aug.random_erasing = False
__C.aug.random_mirror = True
__C.aug.pad = 10
__C.aug.random_crop = True

__C.train = edict()
__C.train.optimizer = 'Adam'
__C.train.lr = 3e-4
__C.train.wd = 5e-4
__C.train.momentum = 0.9
__C.train.step = [80, 180, 300]
__C.train.warmup_epoch = 20
__C.train.warmup_begin_lr = 3e-6
__C.train.factor = 0.1
__C.train.margin = 0.3
__C.train.num_epochs = 400
__C.train.sampler = 'softmax'
__C.train.p_size = 32  # number of person in a single gpu
__C.train.k_size = 4  # number of images per person
__C.train.batch_size = 128
__C.train.triplet_normalize = False

__C.test = edict()
__C.test.batch_size = 128
__C.test.load_path = '/mnt/truenas/scratch/xingyu.liao/DATA/mx-ckpt'

__C.network = edict()
__C.network.depth = 50
__C.network.name = 'Baseline'
__C.network.last_stride = 1
__C.network.gpus = "1"
__C.network.workers = 8

__C.misc = edict()
__C.misc.log_interval = 10
__C.misc.eval_step = 50
__C.misc.save_step = 50


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in __C:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        __C[k][vk] = vv
                else:
                    __C[k] = v
            else:
                raise ValueError("key must exist in configs.py")
