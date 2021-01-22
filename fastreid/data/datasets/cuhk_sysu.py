# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class cuhkSYSU(ImageDataset):
    """CUHK SYSU datasets.

    The dataset is collected from two sources: street snap and movie.
    In street snap, 12,490 images and 6,057 query persons were collected
    with movable cameras across hundreds of scenes while 5,694 images and
    2,375 query persons were selected from movies and TV dramas.

    Dataset statistics:
        - identities: xxx.
        - images: 12936 (train).
    """
    dataset_dir = 'cuhk_sysu'
    dataset_name = "cuhksysu"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, "cropped_images")

        required_files = [self.data_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.data_dir)
        query = []
        gallery = []

        super(cuhkSYSU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'p([-\d]+)_s(\d)')

        data = []
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_0"
            data.append((img_path, pid, camid))

        return data
