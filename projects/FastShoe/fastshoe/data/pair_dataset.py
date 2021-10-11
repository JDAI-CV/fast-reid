# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 18:00:10
# @Author  : zuchen.wang@vipshop.com
# @File    : pair_dataset.py
import os
import random

from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image


class PairDataset(Dataset):
    def __init__(self, img_root: str, pos_folders: list, neg_folders: list, transform=None):
        self.img_root = img_root
        self.pos_folders = pos_folders
        self.neg_folders = neg_folders
        self.transform = transform

    def __len__(self):
        return len(self.pos_folders)

    def __getitem__(self, idx):
        pf, nf = self.pos_folders[idx], self.neg_folders[idx]
        label = 1
        if random.random() < 0.5:
            # generate positive pair
            img_path1, img_path2 = random.sample(pf, 2)
        else:
            # generate negative pair
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
