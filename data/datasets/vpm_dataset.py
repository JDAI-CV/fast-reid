# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import re
import numpy as np

from PIL import Image
import random
from torch.utils.data import Dataset
from .dataset_loader import read_image

__all__ = ['vpmDataset']


class vpmDataset(Dataset):
    """VPM ReID Dataset"""

    def __init__(self, img_items, num_crop=6, crop_ratio=0.5, transform=None, relabel=True):
        self.tfms, self.num_crop, self.crop_ratio, self.relabel = \
                   transform, num_crop, crop_ratio, relabel

        self.pid2label = None
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])  # path
                self.img_items.append((item[0], pid, item[2]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid2label = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items

    @property
    def c(self):
        return len(self.pid2label) if self.pid2label is not None else 0

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)

        img, region_label, num_vis = self.crop_img(img)

        if self.tfms is not None:   
            img = self.tfms(img)
        if self.relabel:            
            pid = self.pid2label[pid]
        return img, pid, camid, region_label, num_vis

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:   
            prefix = 'cuhk'
        else:                       
            prefix = file_path.split('/')[1]
        return prefix + '_' + str(pid)


    def crop_img(self, img):
        # gamma = random.uniform(self.crop_ratio, 1)
        gamma = 1
        w, h = img.size
        crop_h = round(h * gamma)
        # bottom visible
        crop_img = img.crop((0, 0, w, crop_h))

        # Initialize region locator label
        feat_h, feat_w = 24, 8
        region_label = np.zeros(shape=(feat_h, feat_w), dtype=np.int64)

        unit_crop = round(1/self.num_crop/gamma*feat_h)
        for i in range(0, feat_h, unit_crop):
            region_label[i:i+unit_crop, :] = i // unit_crop

        return crop_img, region_label, round(gamma * self.num_crop)


        
