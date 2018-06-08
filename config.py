# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'market'
    height = 384
    width = 192

    # optimization options
    optim = 'SGD'
    max_epoch = 100
    train_batch = 128
    test_batch = 128
    lr = 0.1
    step_size = 60
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    margin = 0.3
    num_instances = 4

    # model options
    model_name = 'ResNetBuilder'
    last_stride = 1

    # miscs
    print_freq = 30
    eval_step = 50
    save_dir = '/DATA/pytorch-ckpt/market'
    gpu = 0, 1
    workers = 10
    start_epoch = 0

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


args = DefaultConfig()
