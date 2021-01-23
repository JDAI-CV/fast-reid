# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['SYSU_mm', ]


@DATASET_REGISTRY.register()
class SYSU_mm(ImageDataset):
    """sysu mm
    """
    dataset_dir = "SYSU-MM01"
    dataset_name = "sysumm01"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []

        file_path_list = ['cam1', 'cam2', 'cam4', 'cam5']

        for file_path in file_path_list:
            camid = self.dataset_name + "_" + file_path
            pid_list = os.listdir(os.path.join(train_path, file_path))
            for pid_dir in pid_list:
                pid = self.dataset_name + "_" + pid_dir
                img_list = glob(os.path.join(train_path, file_path, pid_dir, "*.jpg"))
                for img_path in img_list:
                    data.append([img_path, pid, camid])
        return data
