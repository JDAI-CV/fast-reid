# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VeRiWild(ImageDataset):
    """VeRi-Wild.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Train dataset statistics:
        - identities: 30671.
        - images: 277797.
    """
    dataset_dir = 'VERI-Wild'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_query.txt')
        self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_10000.txt')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info,
        ]
        self.check_before_run(required_files)

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)

        train = self.process_dir(self.train_list)
        query = self.process_dir(self.query_list)
        gallery = self.process_dir(self.gallery_list)

        super(VeRiWild, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_list):
        img_list_lines = open(img_list, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split('/')[0]
            imgid = line.split('/')[1]
            dataset.append((self.imgid2imgpath[imgid], int(vid), int(self.imgid2camid[imgid])))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[1:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path

        assert len(imgid2vid) == len(vehicle_info_lines) - 1
        return imgid2vid, imgid2camid, imgid2imgpath


@DATASET_REGISTRY.register()
class SamllVeRiWild(ImageDataset):
    """VeRi-Wild.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Small test dataset statistics:
        - identities: 3000.
        - images: 41861.
    """
    dataset_dir = 'VERI-Wild'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_3000_query.txt')
        self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_3000.txt')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info,
        ]
        self.check_before_run(required_files)

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)

        train = self.process_dir(self.train_list)
        query = self.process_dir(self.query_list)
        gallery = self.process_dir(self.gallery_list)

        super(SamllVeRiWild, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_list):
        img_list_lines = open(img_list, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split('/')[0]
            imgid = line.split('/')[1]
            dataset.append((self.imgid2imgpath[imgid], int(vid), int(self.imgid2camid[imgid])))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[1:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path

        assert len(imgid2vid) == len(vehicle_info_lines) - 1
        return imgid2vid, imgid2camid, imgid2imgpath


@DATASET_REGISTRY.register()
class MediumVeRiWild(ImageDataset):
    """VeRi-Wild.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Medium test dataset statistics:
        - identities: 5000.
        - images: 69389.
    """
    dataset_dir = 'VERI-Wild'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_5000_query.txt')
        self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_5000.txt')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info,
        ]
        self.check_before_run(required_files)

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)

        train = self.process_dir(self.train_list)
        query = self.process_dir(self.query_list)
        gallery = self.process_dir(self.gallery_list)

        super(MediumVeRiWild, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_list):
        img_list_lines = open(img_list, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split('/')[0]
            imgid = line.split('/')[1]
            dataset.append((self.imgid2imgpath[imgid], int(vid), int(self.imgid2camid[imgid])))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[1:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path

        assert len(imgid2vid) == len(vehicle_info_lines) - 1
        return imgid2vid, imgid2camid, imgid2imgpath


@DATASET_REGISTRY.register()
class LargeVeRiWild(ImageDataset):
    """VeRi-Wild.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Large test dataset statistics:
        - identities: 10000.
        - images: 138517.
    """
    dataset_dir = 'VERI-Wild'
    dataset_url = None

    def __init__(self, root='/home/liuxinchen3/notespace/data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_query.txt')
        self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_10000.txt')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info,
        ]
        self.check_before_run(required_files)

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)

        train = self.process_dir(self.train_list)
        query = self.process_dir(self.query_list)
        gallery = self.process_dir(self.gallery_list)

        super(LargeVeRiWild, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_list):
        img_list_lines = open(img_list, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split('/')[0]
            imgid = line.split('/')[1]
            dataset.append((self.imgid2imgpath[imgid], int(vid), int(self.imgid2camid[imgid])))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[1:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path

        assert len(imgid2vid) == len(vehicle_info_lines) - 1
        return imgid2vid, imgid2camid, imgid2imgpath
