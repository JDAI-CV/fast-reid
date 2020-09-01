# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['VIPeR', ]


@DATASET_REGISTRY.register()
class VIPeR(ImageDataset):
    dataset_dir = "VIPeR"
    dataset_name = "viper"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []

        file_path_list = ['cam_a', 'cam_b']

        for file_path in file_path_list:
            camid = self.dataset_name + "_" + file_path
            img_list = glob(os.path.join(train_path, file_path, "*.bmp"))
            for img_path in img_list:
                img_name = img_path.split('/')[-1]
                pid = self.dataset_name + "_" + img_name.split('_')[0]
                data.append([img_path, pid, camid])

        return data
