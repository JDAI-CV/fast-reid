# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 9:47:53
# @Author  : zuchen.wang@vipshop.com
# @File    : trainer.py
import logging
import os

import torch

from fastreid.data.build import _root
from fastreid.engine import DefaultTrainer
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils import comm
from fastreid.data.transforms import build_transforms
from fastreid.data.build import build_reid_train_loader, build_reid_test_loader
from fastreid.evaluation import PairScoreEvaluator, PairDistanceEvaluator
from projects.FastShoe.fastshoe.data import PairDataset



class PairTrainer(DefaultTrainer):

    _logger = logging.getLogger('fastreid.fastshoe')

    @classmethod
    def build_train_loader(cls, cfg):
        cls._logger.info("Prepare training set")

        transforms = build_transforms(cfg, is_train=True)
        img_root=os.path.join(_root, 'shoe_crop_all_images')
        anno_path=os.path.join(_root, 'labels/1019/1019_clean_train.json')

        datasets = []
        for d in cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(img_root=img_root, anno_path=anno_path, transform=transforms, mode='train')
            if comm.is_main_process():
                dataset.show_train()
            datasets.append(dataset)

        train_set = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
        data_loader = build_reid_train_loader(cfg, train_set=train_set)
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        cls._logger.info("Prepare {} set".format('test' if cfg.eval_only else 'validation'))

        transforms = build_transforms(cfg, is_train=False)
        if dataset_name == 'PairDataset':
            img_root = os.path.join(_root, 'shoe_crop_all_images')
            val_json = os.path.join(_root, 'labels/1019/1019_clean_val.json')
            test_json = os.path.join(_root, 'labels/1019/1019_clean_test.json')

            anno_path, mode = (test_json, 'test') if cfg.eval_only else (val_json, 'val')
            cls._logger.info('Loading {} with {} for {}.'.format(img_root, anno_path, mode))
            test_set = DATASET_REGISTRY.get(dataset_name)(img_root=img_root, anno_path=anno_path, transform=transforms, mode=mode)
            test_set.show_test()

        elif dataset_name == 'ExcelDataset':
            img_root_0830 = os.path.join(_root, 'excel/0830/rotate_shoe_crop_images')
            test_csv_0830 = os.path.join(_root, 'excel/0830/excel_pair_crop.csv')

            img_root_0908 = os.path.join(_root, 'excel/0908/rotate_shoe_crop_images')
            val_csv_0908 = os.path.join(_root, 'excel/0908/excel_pair_crop_val.csv')
            test_csv_0908 = os.path.join(_root, 'excel/0908/excel_pair_crop_test.csv')
            if cfg.eval_only:
                cls._logger.info('Loading {} with {} for test.'.format(img_root_0830, test_csv_0830))
                test_set_0830 = DATASET_REGISTRY.get(dataset_name)(img_root=img_root_0830, anno_path=test_csv_0830, transform=transforms)
                test_set_0830.show_test()

                cls._logger.info('Loading {} with {} for test.'.format(img_root_0908, test_csv_0908))
                test_set_0908 = DATASET_REGISTRY.get(dataset_name)(img_root=img_root_0908, anno_path=test_csv_0908, transform=transforms)
                test_set_0908.show_test()

                test_set = torch.utils.data.ConcatDataset((test_set_0830, test_set_0908))
            else:
                cls._logger.info('Loading {} with {} for validation.'.format(img_root_0908, val_csv_0908))
                test_set = DATASET_REGISTRY.get(dataset_name)(img_root=img_root_0908, anno_path=val_csv_0908, transform=transforms)
                test_set.show_test()
        else:
            raise ValueError("Undefined Dataset!!!")
                
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, PairDistanceEvaluator(cfg, output_dir)
