# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['iLIDS', ]


@DATASET_REGISTRY.register()
class iLIDS(ImageDataset):
    """iLIDS
    """
    dataset_dir = "iLIDS"
    dataset_name = "ilids"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        file_path = os.listdir(train_path)
        for pid_dir in file_path:
            img_file = os.path.join(train_path, pid_dir)
            img_paths = glob(os.path.join(img_file, "*.png"))
            for img_path in img_paths:
                split_path = img_path.split('/')
                pid = self.dataset_name + "_" + split_path[-2]
                camid = self.dataset_name + "_" + split_path[-1].split('_')[0]
                data.append([img_path, pid, camid])
        return data
