# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import re

from PIL import Image
from torch.utils.data import Dataset

__all__ = ['ImageDataset']


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.tfms,self.relabel = transform,relabel

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

        if self.tfms is not None:   img = self.tfms(img)
        if self.relabel:            pid = self.pid2label[pid]
        return img, pid, camid

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:   prefix = 'cuhk'
        else:                       prefix = file_path.split('/')[1]
        return prefix + '_' + str(pid)

