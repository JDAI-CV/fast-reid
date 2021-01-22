# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['SAIVT', ]


@DATASET_REGISTRY.register()
class SAIVT(ImageDataset):
    """SAIVT
    """
    dataset_dir = "SAIVT-SoftBio"
    dataset_name = "saivt"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []

        pid_path = os.path.join(train_path, "cropped_images")
        pid_list = os.listdir(pid_path)

        for pid_name in pid_list:
            pid = self.dataset_name + '_' + pid_name
            img_list = glob(os.path.join(pid_path, pid_name, "*.jpeg"))
            for img_path in img_list:
                img_name = os.path.basename(img_path)
                camid = self.dataset_name + '_' + img_name.split('-')[2]
                data.append([img_path, pid, camid])
        return data
