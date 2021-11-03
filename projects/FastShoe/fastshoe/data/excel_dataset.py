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
class ExcelDataset(ImageDataset):

    _logger = logging.getLogger('fastreid.fastshoe')

    def __init__(self, img_root, anno_path, transform=None, **kwargs):
        self._logger.info('set with {} random seed: 12345'.format(self.__class__.__name__))
        seed_all_rng(12345)

        self.img_root = img_root
        self.anno_path = anno_path
        self.transform = transform

        df = pd.read_csv(self.anno_path) 
        df = df[['内网crop图', '外网crop图', '确认是否撞款']] 
        df['确认是否撞款'] = df['确认是否撞款'].map({'是': 1, '否': 0}) 
        self.df = df

    def __getitem__(self, idx):
        image_inner, image_outer, label = tuple(self.df.loc[idx])

        image_inner_path = os.path.join(self.img_root, image_inner)
        image_outer_path = os.path.join(self.img_root, image_outer)

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
    
    #-------------下面是辅助信息------------------#
    @property
    def num_classes(self):
        return 2
    
    def show_test(self):
        num_pairs = len(self)
        num_images = num_pairs * 2

        headers = ['pairs', 'images']
        csv_results = [[num_pairs, num_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        self._logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
