# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 16:55:30
# @Author  : zuchen.wang@vipshop.com
# @File    : shoe_dataset.py

import logging
import json

from tabulate import tabulate
from termcolor import colored

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class ShoeDataset(ImageDataset):
    def __init__(self, img_dir: str, anno_path: str, **kwargs):
        self.img_dir = img_dir
        self.anno_path = anno_path

        all_data = json.load(open(self.anno_path))
        pos_folders = []
        neg_folders = []
        for data in all_data:
            pos_folders.append(data['positive_img_list'])
            neg_folders.append(data['negative_img_list'])

        assert len(pos_folders) == len(neg_folders), \
            'the len of self.pos_foders should be equal to self.pos_foders'

        super().__init__(pos_folders, neg_folders, None, **kwargs)

    def get_num_pids(self, data):
        return len(data)

    def get_num_cams(self, data):
        return 1

    def parse_data(self, data):
        pids = 0
        imgs = set()
        for info in data:
            pids += 1
            imgs.intersection_update(info)

        return pids, len(imgs)

    def show_train(self):
        num_train_pids, num_train_images = self.parse_data(self.train)
        headers = ['subset', '# folders', '# images']
        csv_results = [['train', num_train_pids, num_train_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )

        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))

    def show_test(self):
        num_query_pids, num_query_images = self.parse_data(self.query)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [['query', num_query_pids, num_query_pids, num_query_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
