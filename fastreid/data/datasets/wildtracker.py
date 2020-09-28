# encoding: utf-8
"""
@author:  wangguanan
@contact: guan.wang0706@gmail.com
"""

import glob
import os

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class WildTrackCrop(ImageDataset):
    """WildTrack.
    Reference:
        WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted Pedestrian Detection
            T. Chavdarova; P. Baqué; A. Maksai; S. Bouquet; C. Jose et al.
    URL: `<https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>`_
    Dataset statistics:
        - identities: 313
        - images: 33979 (train only)
        - cameras: 7
    Args:
        data_path(str): path to WildTrackCrop dataset
        combineall(bool): combine train and test sets as train set if True
    """
    dataset_url = None
    dataset_dir = 'Wildtrack_crop_dataset'
    dataset_name = 'wildtrack'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        self.train_dir = os.path.join(self.dataset_dir, "crop")

        train = self.process_dir(self.train_dir)
        query = []
        gallery = []

        super(WildTrackCrop, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        r"""
        :param dir_path: directory path saving images
        Returns
            data(list) = [img_path, pid, camid]
        """
        data = []
        for dir_name in os.listdir(dir_path):
            img_lists = glob.glob(os.path.join(dir_path, dir_name, "*.png"))
            for img_path in img_lists:
                pid = self.dataset_name + "_" + dir_name
                camid = img_path.split('/')[-1].split('_')[0]
                camid = self.dataset_name + "_" + camid
                data.append([img_path, pid, camid])
        return data
