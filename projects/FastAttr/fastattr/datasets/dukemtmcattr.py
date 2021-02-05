# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re
import mat4py
import numpy as np

from fastreid.data.datasets import DATASET_REGISTRY

from .bases import Dataset


@DATASET_REGISTRY.register()
class DukeMTMCAttr(Dataset):
    """DukeMTMCAttr.

    Reference:
        Lin, Yutian, et al. "Improving person re-identification by attribute and identity learning."
        Pattern Recognition 95 (2019): 151-161.

    URL: `<https://github.com/vana77/DukeMTMC-attribute>`_

    The folder structure should be:
        DukeMTMC-reID/
            bounding_box_train/ # images
            bounding_box_test/ # images
            duke_attribute.mat
    """
    dataset_dir = 'DukeMTMC-reID'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    dataset_name = "dukemtmc"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.test_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.test_dir,
        ]
        self.check_before_run(required_files)

        duke_attr = mat4py.loadmat(osp.join(self.dataset_dir, 'duke_attribute.mat'))['duke_attribute']
        sorted_attrs = sorted(duke_attr['train'].keys())
        sorted_attrs.remove('image_index')
        attr_dict = {i: str(attr) for i, attr in enumerate(sorted_attrs)}

        train = self.process_dir(self.train_dir, duke_attr['train'], sorted_attrs)
        test = val = self.process_dir(self.test_dir, duke_attr['test'], sorted_attrs)

        super(DukeMTMCAttr, self).__init__(train, val, test, attr_dict=attr_dict, **kwargs)

    def process_dir(self, dir_path, annotation, sorted_attrs):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8

            img_index = annotation['image_index'].index(str(pid).zfill(4))
            attrs = np.array([int(annotation[i][img_index]) - 1 for i in sorted_attrs], dtype=np.float32)
            data.append((img_path, attrs))

        return data
