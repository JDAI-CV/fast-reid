# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['AirportALERT', ]


@DATASET_REGISTRY.register()
class AirportALERT(ImageDataset):
    """Airport 

    """
    dataset_dir = "AirportALERT"
    dataset_name = "airport"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)
        self.train_file = os.path.join(self.root, self.dataset_dir, 'filepath.txt')

        required_files = [self.train_file, self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path, self.train_file)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, dir_path, train_file):
        data = []
        with open(train_file, "r") as f:
            img_paths = [line.strip('\n') for line in f.readlines()]

        for path in img_paths:
            split_path = path.split('\\')
            img_path = '/'.join(split_path)
            camid = self.dataset_name + "_" + split_path[0]
            pid = self.dataset_name + "_" + split_path[1]
            img_path = os.path.join(dir_path, img_path)
            # if 11001 <= int(split_path[1]) <= 401999:
            if 11001 <= int(split_path[1]):
                data.append([img_path, pid, camid])

        return data
