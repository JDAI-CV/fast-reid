# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re

from .bases import ImageDataset


class BjStation(ImageDataset):
    dataset_dir = 'beijingStation'

    def __init__(self, root='datasets', return_mask=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.return_mask = return_mask
        self.return_pose = False
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        # self.data_dir = self.dataset_dir
        self.train_summer = osp.join(self.dataset_dir, 'train/train_summer')
        self.train_winter = osp.join(self.dataset_dir, 'train/train_winter')
        # self.train_summer_extra = osp.join(self.dataset_dir, 'train/train_summer_extra')
        # self.train_winter_191204 = osp.join(self.dataset_dir, 'train/train_winter_20191204')
        # self.train_winter_200102 = osp.join(self.dataset_dir, 'train/train_winter_20200102')
        self.query_dir = osp.join(self.dataset_dir, 'benchmark/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'benchmark/gallery')
        # self.query_dir = osp.join(self.dataset_dir, 'benchmark/Crowd_REID/Query')
        # self.gallery_dir = osp.join(self.dataset_dir, 'benchmark/Crowd_REID/Gallery')
        self.mask_dir = osp.join(self.dataset_dir, 'mask')
        self.pose_dir = osp.join(self.dataset_dir, 'pose')

        required_files = [
            # self.train_summer,
            # self.train_winter,
            self.query_dir,
            self.gallery_dir,
            self.mask_dir,
            self.pose_dir
        ]
        self.check_before_run(required_files)

        train = []
        train.extend(self.process_train(self.train_summer))
        train.extend(self.process_train(self.train_winter))
        # train.extend(self.process_train(self.train_summer_extra))
        # train.extend(self.process_train(self.train_winter_191204))
        # train.extend(self.process_train(self.train_winter_200102))
        query, gallery = self.process_test(self.query_dir, self.gallery_dir)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, dir_path):
        img_paths = []
        for d in os.listdir(dir_path):
            img_paths.extend(glob.glob(osp.join(dir_path, d, '*.jpg')))

        pattern = re.compile(r'([-\d]+)_c(\d*)')
        v_paths = []
        for img_path in img_paths:
            pid, camid = map(str, pattern.search(img_path).groups())
            # import shutil
            # root_path = '/'.join(img_path.split('/')[:-1])
            # img_name = img_path.split('/')[-1]
            # new_img_name = img_name.split('v')
            # new_img_name = new_img_name[0]+new_img_name[1]
            # shutil.move(img_path, os.path.join(root_path, new_img_name))
            mask_path = osp.join(self.mask_dir, '/'.join(img_path.split('/')[-3:]))
            pose_path = mask_path[:-3] + 'npy'
            if self.return_mask and self.return_pose:
                v_paths.append([(img_path, mask_path, pose_path), pid, camid])
            elif self.return_mask:
                v_paths.append([(img_path, mask_path), pid, camid])
            elif self.return_pose:
                v_paths.append([(img_path, pose_path), pid, camid])
            else:
                v_paths.append([img_path, pid, camid])

        return v_paths

    def process_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        # gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        gallery_img_paths = []
        id_dirs = os.listdir(gallery_path)
        for d in id_dirs:
            gallery_img_paths.extend(glob.glob(os.path.join(gallery_path, d, '*.jpg')))

        pattern = re.compile(r'([-\d]+)_c(\d*)')
        query_paths = []
        for img_path in query_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # pid = int(pid)
            # if pid == -1: continue  # junk images are just ignored
            if self.return_pose:
                pose_path = osp.join(self.pose_dir, '/'.join(img_path.split('/')[-2:]))
                pose_path = pose_path[:-3] + 'npy'
                query_paths.append([(img_path, pose_path), pid, camid])
            else:
                query_paths.append([img_path, pid, camid])

        gallery_paths = []
        for img_path in gallery_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if self.return_pose:
                pose_path = osp.join(self.pose_dir, '/'.join(img_path.split('/')[-3:]))
                pose_path = pose_path[:-3] + 'npy'
                gallery_paths.append([(img_path, pose_path), pid, camid])
            else:
                gallery_paths.append([img_path, pid, camid])

        return query_paths, gallery_paths
