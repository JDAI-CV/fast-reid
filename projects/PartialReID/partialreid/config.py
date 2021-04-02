# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.config import CfgNode as CN


def add_partialreid_config(cfg):
    _C = cfg

    _C.TEST.DSR = CN({"ENABLED": True})
