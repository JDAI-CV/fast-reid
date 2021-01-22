# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['Shinpuhkan', ]


@DATASET_REGISTRY.register()
class Shinpuhkan(ImageDataset):
    """shinpuhkan
    """
    dataset_dir = "shinpuhkan"
    dataset_name = 'shinpuhkan'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []

        for root, dirs, files in os.walk(train_path):
            img_names = list(filter(lambda x: x.endswith(".jpg"), files))
            # fmt: off
            if len(img_names) == 0: continue
            # fmt: on
            for img_name in img_names:
                img_path = os.path.join(root, img_name)
                split_path = img_name.split('_')
                pid = self.dataset_name + "_" + split_path[0]
                camid = self.dataset_name + "_" + split_path[2]
                data.append((img_path, pid, camid))

        return data
