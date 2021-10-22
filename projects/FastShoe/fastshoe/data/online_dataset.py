# coding: utf-8
import os
import logging

import pandas as pd
from tabulate import tabulate
from termcolor import colored

from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from fastreid.utils.env import seed_all_rng


@DATASET_REGISTRY.register()
class OnlineDataset(ImageDataset):
    def __init__(self, img_dir, anno_path, transform=None, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._logger.info('set with {} random seed: 12345'.format(self.__class__.__name__))
        seed_all_rng(12345)

        self.img_dir = img_dir
        self.anno_path = anno_path
        self.transform = transform

        df = pd.read_csv(self.anno_path) 
        df = df[['内网crop图', '外网crop图', '确认是否撞款']] 
        df['确认是否撞款'] = df['确认是否撞款'].map({'是': 1, '否': 0}) 
        self.df = df

    def __getitem__(self, idx):
        image_inner, image_outer, label = tuple(self.df.loc[idx])

        image_inner_path = os.path.join(self.img_dir, image_inner)
        image_outer_path = os.path.join(self.img_dir, image_outer)

        img1 = read_image(image_inner_path)
        img2 = read_image(image_outer_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'img1': img1,
            'img2': img2,
            'target': label
        }

    def __len__(self):
        return len(self.df)
    
    @property
    def num_classes(self):
        return 2
    
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

    def show_test(self):
        num_query_pids, num_query_images = self.parse_data(self.df['内网crop图'].tolist())

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [['query', num_query_pids, num_query_pids, num_query_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        self._logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
