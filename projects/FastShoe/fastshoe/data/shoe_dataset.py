# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 16:55:30
# @Author  : zuchen.wang@vipshop.com
# @File    : shoe_dataset.py
import os
import json
import random

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset


@DATASET_REGISTRY.register()
class ShoeDataset(ImageDataset):
    def __init__(self, img_dir: str, annotation_json: str, **kwargs):
        self.img_dir = img_dir
        self.annotation_json = annotation_json

        all_data = json.load(open(self.annotation_json))
        pos_folders = []
        neg_folders = []
        for data in all_data:
            pos_folders.append(data['positive_img_list'])
            neg_folders.append(data['negative_img_list'])

        assert len(self.pos_folders) == len(self.neg_folders), \
            'the len of self.pos_foders should be equal to self.pos_foders'

        super().__init__(pos_folders, neg_folders, **kwargs)
