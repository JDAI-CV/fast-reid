# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch
import random
import re

from PIL import Image
from .data_utils import read_image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ReidDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.tfms = transform
        self.relabel = relabel

        self.pid2label = None
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])
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
        if self.tfms is not None:   img = self.tfms(img)
        if self.relabel:            pid = self.pid2label[pid]
        return {
            'images': img,
            'targets': pid,
            'camid': camid
        }

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:
            prefix = 'cuhk'
        else:
            prefix = file_path.split('/')[1]
        return prefix + '_' + str(pid)
