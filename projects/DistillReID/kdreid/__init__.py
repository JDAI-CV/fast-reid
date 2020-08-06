# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from .config import add_kdreid_config, add_shufflenet_config
from .kd_trainer import KDTrainer
from .modeling import build_shufflenetv2_backbone