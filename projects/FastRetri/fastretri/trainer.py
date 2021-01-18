# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging

from torchvision import transforms as T

from fastreid.data import build_reid_train_loader, build_reid_test_loader
from fastreid.data.transforms import ToTensor
from fastreid.engine import DefaultTrainer

from .retri_evaluator import RetriEvaluator


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger("fastreid.dml_dataset")
        logger.info("Prepare training set")

        mapper = []
        if cfg.INPUT.SIZE_TRAIN[0] > 0:
            if len(cfg.INPUT.SIZE_TRAIN) == 1:
                resize = cfg.INPUT.SIZE_TRAIN[0]
            else:
                resize = cfg.INPUT.SIZE_TRAIN
            mapper.append(T.Resize(resize, interpolation=3))

        if cfg.INPUT.CJ.ENABLED:
            cj_params = [
                cfg.INPUT.CJ.BRIGHTNESS,
                cfg.INPUT.CJ.CONTRAST,
                cfg.INPUT.CJ.SATURATION,
                cfg.INPUT.CJ.HUE
            ]
            mapper.append(T.ColorJitter(*cj_params))

        mapper.extend([
            T.RandomResizedCrop(size=cfg.INPUT.CROP_SIZE, scale=cfg.INPUT.SCALE,
                                ratio=cfg.INPUT.RATIO, interpolation=3),
            T.RandomHorizontalFlip(),
            ToTensor(),
        ])
        return build_reid_train_loader(cfg, mapper=T.Compose(mapper))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_test_loader`.
        Overwrite it if you'd like a different data loader.
        """

        mapper = []
        if cfg.INPUT.SIZE_TEST[0] > 0:
            if len(cfg.INPUT.SIZE_TEST) == 1:
                resize = cfg.INPUT.SIZE_TEST[0]
            else:
                resize = cfg.INPUT.SIZE_TEST
            mapper.append(T.Resize(resize, interpolation=3))

        mapper.extend([
            T.CenterCrop(size=cfg.INPUT.CROP_SIZE),
            ToTensor(),
        ])
        return build_reid_test_loader(cfg, dataset_name, mapper=T.Compose(mapper))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        return data_loader, RetriEvaluator(cfg, num_query, output_dir)
