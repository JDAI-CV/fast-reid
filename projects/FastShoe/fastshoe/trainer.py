# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 9:47:53
# @Author  : zuchen.wang@vipshop.com
# @File    : trainer.py
import logging
import os

from fastreid.data.build import _root
from fastreid.engine import DefaultTrainer
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils import comm
from fastreid.data.transforms import build_transforms
from fastreid.data.build import build_reid_train_loader, build_reid_test_loader
from fastreid.evaluation.pair_evaluator import PairEvaluator

from projects.FastShoe.fastshoe.data import PairDataset


class PairTrainer(DefaultTrainer):

    shoe_img_dir = os.path.join(_root, 'shoe_crop_all_images')
    shoe_anno_dir = os.path.join(_root, 'labels/0930')

    excel_img_dir = os.path.join(_root, 'excel/shoe_crop_images')
    excel_anno_path = os.path.join(_root, 'excel/excel_pair_crop.csv')
    # excel_anno_path = os.path.join(_root, 'excel/temp_test.csv')

    @classmethod
    def build_train_loader(cls, cfg):
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")

        pos_folder_list, neg_folder_list = list(), list()
        for d in cfg.DATASETS.NAMES:
            data = DATASET_REGISTRY.get(d)(img_dir=cls.shoe_img_dir,
                                           anno_path=os.path.join(cls.shoe_anno_dir, '0930_clean_train.json'))
            if comm.is_main_process():
                data.show_train()
            pos_folder_list.extend(data.train)
            neg_folder_list.extend(data.query)

        transforms = build_transforms(cfg, is_train=True)
        train_set = PairDataset(img_root=cls.shoe_img_dir,
                                pos_folders=pos_folder_list, neg_folders=neg_folder_list, transform=transforms, mode='train')
        data_loader = build_reid_train_loader(cfg, train_set=train_set)
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        transforms = build_transforms(cfg, is_train=False)
        if dataset_name == 'ShoeDataset':
            if cfg.eval_only:
                mode = 'test'
                anno_path = os.path.join(cls.shoe_anno_dir, '0930_clean_test.json')
            else:
                mode = 'val'
                anno_path = os.path.join(cls.shoe_anno_dir, '0930_clean_val.json')

            data = DATASET_REGISTRY.get(dataset_name)(img_dir=cls.shoe_img_dir, anno_path=anno_path)
            test_set = PairDataset(img_root=cls.shoe_img_dir,
                                   pos_folders=data.train, neg_folders=data.query, transform=transforms, mode=mode)
        elif dataset_name == 'OnlineDataset':
            test_set = DATASET_REGISTRY.get(dataset_name)(img_dir=cls.excel_img_dir,
                                                          anno_path=cls.excel_anno_path,
                                                          transform=transforms)
        if comm.is_main_process():
            if dataset_name == 'ShoeDataset':
                data.show_test()
            else:
                test_set.show_test()

        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, PairEvaluator(cfg, output_dir)
