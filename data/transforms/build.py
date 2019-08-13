# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .transforms import *
from fastai.vision.transform import *


def build_transforms(cfg):
    "Utility func to easily create a list of flip, rotate, `zoom`, warp, lighting transforms."
    res = []
    if cfg.INPUT.DO_FLIP:   res.append(flip_lr(p=cfg.INPUT.FLIP_PROB))
    if cfg.INPUT.DO_PAD:    res.extend(rand_pad(padding=cfg.INPUT.PADDING,
                                                size=cfg.INPUT.SIZE_TRAIN,
                                                mode=cfg.INPUT.PADDING_MODE))
    if cfg.INPUT.DO_LIGHTING:
        res.append(brightness(change=(0.5*(1-cfg.INPUT.MAX_LIGHTING), 0.5*(1+cfg.INPUT.MAX_LIGHTING)), p=cfg.INPUT.P_LIGHTING))
        res.append(contrast(scale=(1-cfg.INPUT.MAX_LIGHTING, 1/(1-cfg.INPUT.MAX_LIGHTING)), p=cfg.INPUT.P_LIGHTING))
    res.append(RandomErasing())
    #       train                   , valid
    return (res, [crop_pad()])