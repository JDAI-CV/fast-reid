# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['CAVIARa', ]


@DATASET_REGISTRY.register()
class CAVIARa(ImageDataset):
    """CAVIARa
    """
    dataset_dir = "CAVIARa"
    dataset_name = "caviara"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []

        img_list = glob(os.path.join(train_path, "*.jpg"))
        for img_path in img_list:
            img_name = img_path.split('/')[-1]
            pid = self.dataset_name + "_" + img_name[:4]
            camid = self.dataset_name + "_cam0"
            data.append([img_path, pid, camid])

        return data
