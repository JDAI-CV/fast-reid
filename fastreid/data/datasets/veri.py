# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VeRi(ImageDataset):
    """VeRi.

    Reference:
        Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = 'veri'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            data.append((img_path, pid, camid))

        return data
