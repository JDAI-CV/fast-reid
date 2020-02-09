# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        res.append(T.Resize(cfg.INPUT.SIZE_TRAIN))
        if cfg.INPUT.DO_FLIP:
            res.append(T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB))
        if cfg.INPUT.DO_PAD:
            res.extend([T.Pad(cfg.INPUT.PADDING, padding_mode=cfg.INPUT.PADDING_MODE),
                        T.RandomCrop(cfg.INPUT.SIZE_TRAIN)])
        # res.append(random_angle_rotate())
        # res.append(do_color())
        # res.append(T.ToTensor())  # to slow
        if cfg.INPUT.RE.DO:
            res.append(RandomErasing(probability=cfg.INPUT.RE.PROB, mean=cfg.INPUT.RE.MEAN))
        if cfg.INPUT.CUTOUT.DO:
            res.append(Cutout(probability=cfg.INPUT.CUTOUT.PROB, size=cfg.INPUT.CUTOUT.SIZE,
                              mean=cfg.INPUT.CUTOUT.MEAN))
    else:
        res.append(T.Resize(cfg.INPUT.SIZE_TEST))
        # res.append(T.ToTensor())
    return T.Compose(res)
