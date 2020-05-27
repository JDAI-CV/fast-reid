# encoding: utf-8

"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

import glob
import os
import os.path as osp
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PartialREID', 'PartialiLIDS', 'OccludedREID']


def process_test(query_path, gallery_path):
    query_img_paths = glob.glob(os.path.join(query_path, '*.jpg'))
    gallery_img_paths = glob.glob(os.path.join(gallery_path, '*.jpg'))
    query_paths = []
    pattern = re.compile(r'([-\d]+)_(\d*)')
    for img_path in query_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        query_paths.append([img_path, pid, camid])
    gallery_paths = []
    for img_path in gallery_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        gallery_paths.append([img_path, pid, camid])
    return query_paths, gallery_paths


@DATASET_REGISTRY.register()
class PartialREID(ImageDataset):
    def __init__(self, root='datasets', ):
        self.root = root

        self.query_dir = osp.join(self.root, 'PartialREID/query')
        self.gallery_dir = osp.join(self.root, 'PartialREID/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class PartialiLIDS(ImageDataset):
    def __init__(self, root='datasets', ):
        self.root = root

        self.query_dir = osp.join(self.root, 'PartialiLIDS/query')
        self.gallery_dir = osp.join(self.root, 'PartialiLIDS/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class OccludedREID(ImageDataset):
    def __init__(self, root='datasets', ):
        self.root = root

        self.query_dir = osp.join(self.root, 'OccludedREID/query')
        self.gallery_dir = osp.join(self.root, 'OccludedREID/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)
