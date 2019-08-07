# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .transforms import *
from fastai.vision.transform import *


def build_transforms(cfg):
    # do_flip:bool=True, max_rotate:float=10., max_zoom:float=1.1,
    #                max_lighting:float=0.2, max_warp:float=0.2, p_affine:float=0.75,
    #                p_lighting:float=0.75, xtra_tfms:Optional[Collection[Transform]]=None):
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