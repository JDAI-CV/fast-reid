# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os

import bcolz
import numpy as np

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["CPLFW", "VGG2_FP", "AgeDB_30", "CALFW", "CFP_FF", "CFP_FP", "LFW"]


@DATASET_REGISTRY.register()
class CPLFW(ImageDataset):
    dataset_dir = "faces_emore_val"
    dataset_name = "cplfw"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]

        self.check_before_run(required_files)

        carray = bcolz.carray(rootdir=os.path.join(self.dataset_dir, self.dataset_name), mode='r')
        is_same = np.load(os.path.join(self.dataset_dir, "{}_list.npy".format(self.dataset_name)))

        self.carray = carray
        self.is_same = is_same

        super().__init__([], [], [], **kwargs)


@DATASET_REGISTRY.register()
class VGG2_FP(CPLFW):
    dataset_name = "vgg2_fp"


@DATASET_REGISTRY.register()
class AgeDB_30(CPLFW):
    dataset_name = "agedb_30"


@DATASET_REGISTRY.register()
class CALFW(CPLFW):
    dataset_name = "calfw"


@DATASET_REGISTRY.register()
class CFP_FF(CPLFW):
    dataset_name = "cfp_ff"


@DATASET_REGISTRY.register()
class CFP_FP(CPLFW):
    dataset_name = "cfp_fp"


@DATASET_REGISTRY.register()
class LFW(CPLFW):
    dataset_name = "lfw"
