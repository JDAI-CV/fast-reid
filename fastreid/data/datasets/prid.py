# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PRID', ]


@DATASET_REGISTRY.register()
class PRID(ImageDataset):
    """PRID
    """
    dataset_dir = "prid_2011"
    dataset_name = 'prid'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir, 'slim_train')

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        for root, dirs, files in os.walk(train_path):
            for img_name in filter(lambda x: x.endswith('.png'), files):
                img_path = os.path.join(root, img_name)
                pid = self.dataset_name + '_' + root.split('/')[-1].split('_')[1]
                camid = self.dataset_name + '_' + img_name.split('_')[0]
                data.append([img_path, pid, camid])
        return data
