# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from fastreid.config import CfgNode as CN


def add_attr_config(cfg):
    _C = cfg

    _C.MODEL.LOSSES.BCE = CN()
    _C.MODEL.LOSSES.BCE.WEIGHT_ENABLED = True
    _C.MODEL.LOSSES.BCE.SCALE = 1.

    _C.TEST.THRES = 0.5
