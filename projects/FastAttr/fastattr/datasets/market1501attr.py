# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import mat4py
import numpy as np

from fastreid.data.datasets import DATASET_REGISTRY

from .bases import Dataset


@DATASET_REGISTRY.register()
class Market1501Attr(Dataset):
    """Market1501Attr.

    Reference:
        Lin, Yutian, et al. "Improving person re-identification by attribute and identity learning."
        Pattern Recognition 95 (2019): 151-161.

    URL: `<https://github.com/vana77/Market-1501_Attribute>`_

    The folder structure should be:
        Market-1501-v15.09.15/
            bounding_box_train/ # images
            bounding_box_test/ # images
            market_attribute.mat
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.test_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.test_dir,
        ]
        self.check_before_run(required_files)

        market_attr = mat4py.loadmat(osp.join(self.data_dir, 'market_attribute.mat'))['market_attribute']
        sorted_attrs = sorted(market_attr['train'].keys())
        sorted_attrs.remove('image_index')
        attr_dict = {i: str(attr) for i, attr in enumerate(sorted_attrs)}

        train = self.process_dir(self.train_dir, market_attr['train'], sorted_attrs)
        test = val = self.process_dir(self.test_dir, market_attr['test'], sorted_attrs)

        super(Market1501Attr, self).__init__(train, val, test, attr_dict=attr_dict, **kwargs)

    def process_dir(self, dir_path, annotation, sorted_attrs):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid == 0:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6

            img_index = annotation['image_index'].index(str(pid).zfill(4))
            attrs = np.array([int(annotation[i][img_index])-1 for i in sorted_attrs], dtype=np.float32)
            data.append((img_path, attrs))

        return data
