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
        do_re = cfg.INPUT.RE.ENABLED
        re_prob = cfg.INPUT.RE.PROB
        re_mean = cfg.INPUT.RE.MEAN
        # cutout
        do_cutout = cfg.INPUT.CUTOUT.ENABLED
        cutout_prob = cfg.INPUT.CUTOUT.PROB
        cutout_size = cfg.INPUT.CUTOUT.SIZE
        cutout_mean = cfg.INPUT.CUTOUT.MEAN
        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_re:
            res.append(RandomErasing(probability=re_prob, mean=re_mean))
        if do_cutout:
            res.append(Cutout(probability=cutout_prob, size=cutout_size,
                              mean=cutout_mean))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=3))
    res.append(ToTensor())
    return T.Compose(res)
