# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['SenseReID', ]


@DATASET_REGISTRY.register()
class SenseReID(ImageDataset):
    """Sense reid
    """
    dataset_dir = "SenseReID"
    dataset_name = "senseid"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        file_path_list = ['test_gallery', 'test_prob']

        for file_path in file_path_list:
            sub_file = os.path.join(train_path, file_path)
            img_name = glob(os.path.join(sub_file, "*.jpg"))
            for img_path in img_name:
                img_name = img_path.split('/')[-1]
                img_info = img_name.split('_')
                pid = self.dataset_name + "_" + img_info[0]
                camid = self.dataset_name + "_" + img_info[1].split('.')[0]
                data.append([img_path, pid, camid])
        return data
