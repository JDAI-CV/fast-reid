# -*- coding: utf-8 -*-

import os
import logging
import json
import random

import pandas as pd
from tabulate import tabulate
from termcolor import colored

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.data_utils import read_image
from fastreid.utils.env import seed_all_rng


@DATASET_REGISTRY.register()
class PairDataset(ImageDataset):

    _logger = logging.getLogger('fastreid.fastshoe')

    def __init__(self, img_root: str, anno_path: str, transform=None, mode: str = 'train'):

        assert mode in ('train', 'val', 'test'), self._logger.info(
                '''mode should the one of ('train', 'val', 'test')''')
        self.mode = mode
        if self.mode != 'train':
            self._logger.info('set {} with {} random seed: 12345'.format(self.mode, self.__class__.__name__))
            seed_all_rng(12345)

        self.img_root = img_root
        self.anno_path = anno_path
        self.transform = transform

        all_data = json.load(open(self.anno_path))
        pos_folders = []
        neg_folders = []
        for data in all_data:
            pos_folders.append(data['positive_img_list'])
            neg_folders.append(data['negative_img_list'])

        assert len(pos_folders) == len(neg_folders), self._logger.error('the len of self.pos_foders should be equal to self.pos_foders')
        self.pos_folders = pos_folders
        self.neg_folders = neg_folders

    def __len__(self):
        if self.mode == 'test':
            return len(self.pos_folders) * 10

        return len(self.pos_folders)

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx = int(idx / 10)
        
        pf = self.pos_folders[idx]
        nf = self.neg_folders[idx]

        label = 1
        if random.random() < 0.5:
            # generate positive pair
            img_path1, img_path2 = random.sample(pf, 2)
        else:
            # generate negative pair
            label = 0
            img_path1, img_path2 = random.choice(pf), random.choice(nf)

        img_path1 = os.path.join(self.img_root, img_path1)
        img_path2 = os.path.join(self.img_root, img_path2)

        img1 = read_image(img_path1)
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
    def num_folders(self):
        return len(self)
    
    @property   
    def num_pos_images(self):
        return sum([len(x) for x in self.pos_folders])

    @property   
    def num_neg_images(self):
        return sum([len(x) for x in self.neg_folders])

    def describe(self):
        headers = ['subset', 'folders', 'pos images', 'neg images']
        csv_results = [[self.mode, self.num_folders, self.num_pos_images, self.num_neg_images]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )

        self._logger.info(f"=> Loaded {self.__class__.__name__}: \n" + colored(table, "cyan"))

    def show_train(self):
        return self.describe()

    def show_test(self):
        return self.describe()
