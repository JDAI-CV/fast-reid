# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .baseline_model import ResNetBuilder


def get_baseline_model(num_classes, last_stride=1, model_path='/DATA/model_zoo/resnet50-19c8e357.pth'):
    model = ResNetBuilder(num_classes, last_stride, model_path)
    optim_policy = model.get_optim_policy()
    return model, optim_policy
