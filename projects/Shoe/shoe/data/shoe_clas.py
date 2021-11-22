# -*- coding: utf-8 -*-

import os
import logging
import json
import random
import math

import numpy as np
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
from tabulate import tabulate
from termcolor import colored
from PIL import Image

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.data_utils import read_image
from fastreid.utils.env import seed_all_rng
from .augment import augment_pos_image
from .shoe import ShoeDataset


@DATASET_REGISTRY.register()
class ShoeClasDataset(ImageDataset):

    def __init__(self, img_root: str, anno_path: str, transform=None, mode: str = 'train'):
        super(ShoeClassDataset, self).__init__(img_root, anno_path, transform, mode)

        self.pos_folders = []
        for data in self.all_data:
            if len(data['positive_img_list']) >= 1: 
                self.pos_folders.append(data['positive_img_list'])

        if self.mode == 'val':
            # for validation in train phase: 
            # use 2 sample per folder(class) since 1 is to little and more is compute expensive
            self.num_images = len(self.pos_folders) * 2

        self.image_paths = []
        self.image_labels = []
        for idx, folder in enumerate(self.pos_folders):
            for img_path in folder:
                self.image_paths.append(img_path)
                self.image_labels.append(idx)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        pos_aug_ratio = 0.5
        if self.mode == 'val':
            pos_aug_ratio = 1
            idx = idx % len(self)
            
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]

        img_path = os.path.join(self.img_root, img_path)
        img = read_image(img_path)

        if random.random() <= pos_aug_ratio:
            img = augment_pos_image(img)

        if self.transform:
            img = self.transform(img)

        return {
            'images': img,
            'targets': label
        }

    #-------------下面是辅助信息------------------#
    @property
    def num_classes(self):
        return len(self.pos_folders)
    
    def describe(self):
        headers = ['subset', 'classes', 'images']
        csv_results = [[self.mode, self.num_classes, self.num_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )

        self._logger.info(f"=> Loaded {self.__class__.__name__}: \n" + colored(table, "cyan"))
