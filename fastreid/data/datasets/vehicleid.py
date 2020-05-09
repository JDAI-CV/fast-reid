# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import os.path as osp
import random

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VehicleID(ImageDataset):
    """VehicleID.

    Reference:
        Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.

    URL: `<https://pkuml.org/resources/pku-vehicleid.html>`_

    Train dataset statistics:
        - identities: 13164.
        - images: 113346.
    """
    dataset_dir = 'vehicleid'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_13164.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(VehicleID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split(' ')[1]
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, imgid + '.jpg')
            dataset.append((img_path, int(vid), int(imgid)))

        random.shuffle(dataset)
        vid_container = set()
        if is_train:
            return dataset
        else:
            query = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    query.append(sample)

            return query, dataset


@DATASET_REGISTRY.register()
class SmallVehicleID(ImageDataset):
    """VehicleID.

    Reference:
        Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.

    URL: `<https://pkuml.org/resources/pku-vehicleid.html>`_

    Small test dataset statistics:
        - identities: 800.
        - images: 6493.
    """
    dataset_dir = 'vehicleid'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_800.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(SmallVehicleID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split(' ')[1]
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, imgid + '.jpg')
            dataset.append((img_path, int(vid), int(imgid)))

        random.shuffle(dataset)
        vid_container = set()
        if is_train:
            return dataset
        else:
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    gallery.append(sample)
                else:
                    query.append(sample)

            return query, gallery


@DATASET_REGISTRY.register()
class MediumVehicleID(ImageDataset):
    """VehicleID.

    Reference:
        Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.

    URL: `<https://pkuml.org/resources/pku-vehicleid.html>`_

    Medium test dataset statistics:
        - identities: 1600.
        - images: 13377.
    """
    dataset_dir = 'vehicleid'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_1600.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(MediumVehicleID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split(' ')[1]
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, imgid + '.jpg')
            dataset.append((img_path, int(vid), int(imgid)))

        random.shuffle(dataset)
        vid_container = set()
        if is_train:
            return dataset
        else:
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    gallery.append(sample)
                else:
                    query.append(sample)

            return query, gallery


@DATASET_REGISTRY.register()
class LargeVehicleID(ImageDataset):
    """VehicleID.

    Reference:
        Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.

    URL: `<https://pkuml.org/resources/pku-vehicleid.html>`_

    Large test dataset statistics:
        - identities: 2400.
        - images: 19777.
    """
    dataset_dir = 'vehicleid'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_2400.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(LargeVehicleID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split(' ')[1]
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, imgid + '.jpg')
            dataset.append((img_path, int(vid), int(imgid)))

        random.shuffle(dataset)
        vid_container = set()
        if is_train:
            return dataset
        else:
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    gallery.append(sample)
                else:
                    query.append(sample)

            return query, gallery
