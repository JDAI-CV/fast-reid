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
        size_train = cfg.INPUT.SIZE_TRAIN
        # filp lr
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB
        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE
        # random erasing
        do_re = cfg.INPUT.RE.DO
        re_prob = cfg.INPUT.RE.PROB
        re_mean = cfg.INPUT.RE.MEAN
        res.append(T.Resize(size_train))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_re:
            res.append(RandomErasing(probability=re_prob, mean=re_mean))
        # if cfg.INPUT.CUTOUT.DO:
        #     res.append(Cutout(probability=cfg.INPUT.CUTOUT.PROB, size=cfg.INPUT.CUTOUT.SIZE,
        #                       mean=cfg.INPUT.CUTOUT.MEAN))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test))
    return T.Compose(res)
