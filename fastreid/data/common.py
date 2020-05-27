# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])
                self.img_items.append((item[0], pid, item[2]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
        return {
            'images': img,
            'targets': pid,
            'camid': camid,
            'img_path': img_path
        }

    @staticmethod
    def get_pids(file_path, pid):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:
            prefix = 'cuhk'
        else:
            prefix = file_path.split('/')[1]
        return prefix + '_' + str(pid)

    def update_pid_dict(self, pid_dict):
        self.pid_dict = pid_dict
