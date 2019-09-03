# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .baseline import Baseline
from .losses import reidLoss


def build_model(cfg, num_classes) -> nn.Module:
    model = Baseline(
        cfg.MODEL.BACKBONE, 
        num_classes, 
        cfg.MODEL.LAST_STRIDE, 
        cfg.MODEL.WITH_IBN, 
        cfg.MODEL.GCB, 
        cfg.MODEL.STAGE_WITH_GCB, 
        cfg.MODEL.PRETRAIN, 
        cfg.MODEL.PRETRAIN_PATH)
    return model
