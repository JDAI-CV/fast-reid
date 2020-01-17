# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import glob
import re

import os.path as osp

from .bases import ImageDataset
import warnings


class SeFresh(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = '7fresh'

    def __init__(self, timeline, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, timeline)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, '7fresh_crop_img_true')

        #########################################################################
        # 
        # import shutil
        # import glob
        # import numpy as np
        # import os
        # id_folders = os.listdir(self.train_dir)
        # for i in id_folders:
        #     all_imgs = glob.glob(os.path.join(self.train_dir, i, '*.jpg'))
        #     query_imgs = np.random.choice(all_imgs, 2, replace=False)
        #     for j in query_imgs:
        #         shutil.move(j, os.path.join(self.data_dir, 'query', j.split('/')[-1]))
        # all_imgs= glob.glob(os.path.join(self.data_dir, 'query', '*.jpg'))
        # for i in all_imgs:
        #     name = i.split('/')[-1]
        #     folder = i.split('/')[-1].split('_')[0]
            # shutil.copy(i, os.path.join(self.train_dir, folder, name))
        
        #########################################################################
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, '7fresh_crop_img_true')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_train(self.train_dir, relabel=True)
        query, gallery = self.process_test(self.query_dir, self.gallery_dir)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, dir_path, query=False, relabel=False):
        if query:   
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:       
            img_paths = []
            for d in os.listdir(dir_path):
                img_paths.extend(glob.glob(osp.join(dir_path, d, '*.jpg')))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        if relabel:
            pid_container = set()
            for img_path in img_paths:
                pid, _ = pattern.search(img_path).groups()
                if pid == -1:
                    continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = pattern.search(img_path).groups()
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data

    def process_test(self, query_path, gallery_path):
        query_imgs = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_imgs = []
        for d in os.listdir(gallery_path):
            gallery_imgs.extend(glob.glob(osp.join(gallery_path, d, '*.jpg')))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in query_imgs:
            pid, _ = pattern.search(img_path).groups()
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        query_data = []
        gallery_data = []
        for img_path in query_imgs:
            pid, camid = pattern.search(img_path).groups()
            pid = pid2label[pid]
            query_data.append((img_path, pid, int(camid)))
        for img_path in gallery_imgs:
            pid, camid = pattern.search(img_path).groups()
            if pid in pid2label:
                pid = pid2label[pid]
            else:
                pid = -1
            gallery_data.append((img_path, pid, int(camid)))

        return query_data, gallery_data

