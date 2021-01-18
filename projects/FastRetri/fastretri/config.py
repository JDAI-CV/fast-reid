# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""


def add_retri_config(cfg):
    _C = cfg

    _C.INPUT.CROP_SIZE = 224
    _C.INPUT.SCALE = (0.16, 1)
    _C.INPUT.RATIO = (3./4., 4./3.)

    _C.TEST.RECALLS = [1, 2, 4, 8, 16, 32]
