# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['LPW', ]


@DATASET_REGISTRY.register()
class LPW(ImageDataset):
    """LPW
    """
    dataset_dir = "pep_256x128/data_slim"
    dataset_name = "lpw"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []

        file_path_list = ['scen1', 'scen2', 'scen3']

        for scene in file_path_list:
            cam_list = os.listdir(os.path.join(train_path, scene))
            for cam in cam_list:
                camid = self.dataset_name + "_" + cam
                pid_list = os.listdir(os.path.join(train_path, scene, cam))
                for pid_dir in pid_list:
                    img_paths = glob(os.path.join(train_path, scene, cam, pid_dir, "*.jpg"))
                    for img_path in img_paths:
                        pid = self.dataset_name + "_" + scene + "-" + pid_dir
                        data.append([img_path, pid, camid])
        return data
