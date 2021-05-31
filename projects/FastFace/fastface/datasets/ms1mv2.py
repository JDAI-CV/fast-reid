# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset


@DATASET_REGISTRY.register()
class MS1MV2(ImageDataset):
    dataset_dir = "MS_Celeb_1M"
    dataset_name = "ms1mv2"

    def __init__(self, root="datasets", **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        train = self.process_dirs()[:10000]
        super().__init__(train, [], [], **kwargs)

    def process_dirs(self):
        train_list = []

        fid_list = os.listdir(self.dataset_dir)

        for fid in fid_list:
            all_imgs = glob.glob(os.path.join(self.dataset_dir, fid, "*.jpg"))
            for img_path in all_imgs:
                train_list.append([img_path, self.dataset_name + '_' + fid, '0'])

        return train_list
