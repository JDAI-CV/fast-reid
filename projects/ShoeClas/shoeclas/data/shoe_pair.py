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
from .augment import augment_pos_image, augment_neg_image
from .shoe import ShoeDataset


@DATASET_REGISTRY.register()
class ShoePairDataset(ShoeDataset):

    def __init__(self, img_root: str, anno_path: str, transform=None, mode: str = 'train'):
        super(ShoePairDataset, self).__init__(img_root, anno_path, transform, mode)

        self.pos_folders = []
        self.neg_folders = []
        for data in self.all_data:
            if len(data['positive_img_list']) >= 2 and len(data['negative_img_list']) >= 1:
                self.pos_folders.append(data['positive_img_list'])
                self.neg_folders.append(data['negative_img_list'])

    def __len__(self):
        return len(self.pos_folders)

    def __getitem__(self, idx):
        pos_aug_ratio = 0.5
        neg_aug_ratio = 0
        
        pf = self.pos_folders[idx]
        nf = self.neg_folders[idx]

        label = 1
        use_pseudo = False
        if self.mode == 'train':
            if random.random() < 0.5:
                # generate positive pair
                if random.random() < pos_aug_ratio:
                    use_pseudo = True
                else:
                    img_path1, img_path2 = random.sample(pf, 2)
            else:
                # generate negative pair
                label = 0
                if random.random() < neg_aug_ratio:
                    use_pseudo = True
                else:
                    img_path1, img_path2 = random.choice(pf), random.choice(nf)

            if use_pseudo:
                img_path1 = random.choice(pf)
        else:
            if random.random() < 0.5:
                img_path1, img_path2 = random.sample(pf, 2)
            else:
                label = 0
                img_path1, img_path2 = random.choice(pf), random.choice(nf)


        img_path1 = os.path.join(self.img_root, img_path1)
        img1 = read_image(img_path1)

        if use_pseudo:
            if label == 1:
                img2 = augment_pos_image(img1)
            else:
                img2 = augment_neg_image(self.img_root, nf, img1)
        else:
            img_path2 = os.path.join(self.img_root, img_path2)
            img2 = read_image(img_path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'img1': img1,
            'img2': img2,
            'target': label
        }

    #-------------下面是辅助信息------------------#
    @property
    def num_classes(self):
        return 2
    
    @property   
    def num_pos_images(self):
        return sum([len(x) for x in self.pos_folders])

    @property   
    def num_neg_images(self):
        return sum([len(x) for x in self.neg_folders])

    def describe(self):
        headers = ['subset', 'folders', 'pos images', 'neg images']
        csv_results = [[self.mode, len(self), self.num_pos_images, self.num_neg_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )

        self._logger.info(f"=> Loaded {self.__class__.__name__}: \n" + colored(table, "cyan"))
