# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PKU', ]


@DATASET_REGISTRY.register()
class PKU(ImageDataset):
    """PKU
    """
    dataset_dir = "PKUv1a_128x48"
    dataset_name = 'pku'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        img_paths = glob(os.path.join(train_path, "*.png"))

        for img_path in img_paths:
            split_path = img_path.split('/')
            img_info = split_path[-1].split('_')
            pid = self.dataset_name + "_" + img_info[0]
            camid = self.dataset_name + "_" + img_info[1]
            data.append([img_path, pid, camid])
        return data
