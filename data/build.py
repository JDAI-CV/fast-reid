# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import re

from fastai.vision import *
from .transforms import RandomErasing
from .samplers import RandomIdentitySampler


def get_data_bunch(cfg):
    ds_tfms = (
        [flip_lr(p=0.5),
         *rand_pad(padding=cfg.INPUT.PADDING, size=cfg.INPUT.SIZE_TRAIN, mode='zeros'),
         RandomErasing()
         ],
        None
    )

    def _process_dir(dir_path):
        img_paths = []
        if 'beijingStation' in dir_path:
            id_dirs = os.listdir(dir_path)
            for d in id_dirs:
                img_paths.extend(glob.glob(os.path.join(dir_path, d, '*.jpg')))
        else:
            img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        v_paths = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            v_paths.append([img_path, pid, camid])
        return v_paths

    market_train_path = 'datasets/Market-1501-v15.09.15/bounding_box_train'
    duke_train_path = 'datasets/DukeMTMC-reID/bounding_box_train'
    cuhk03_train_path = 'datasets/cuhk03/'
    bjStation_train_path = 'datasets/beijingStation/20190720/train'

    query_path = 'datasets/Market-1501-v15.09.15/query'
    gallery_path = 'datasets/Market-1501-v15.09.15/bounding_box_test'

    train_img_names = _process_dir(market_train_path) + _process_dir(duke_train_path) + _process_dir(bjStation_train_path)
    # train_img_names = CUHK03().train
    train_names = [i[0] for i in train_img_names]

    query_names = _process_dir(query_path)
    gallery_names = _process_dir(gallery_path)
    test_fnames = []
    test_labels = []
    for i in query_names+gallery_names:
        test_fnames.append(i[0])
        test_labels.append(i[1:])

    def get_labels(file_path):
        """ Suitable for muilti-dataset training """
        # if 'cuhk03' in file_path:
            # prefix = 'cuhk'
            # pid = file_path.split('/')[-1].split('_')[1]
        # else:
        prefix = file_path.split('/')[1]
        pat = re.compile(r'([-\d]+)_c(\d)')
        pid, _ = pat.search(file_path).groups()
        return prefix + '_' + pid

    data_bunch = ImageDataBunch.from_name_func('datasets', train_names, label_func=get_labels, valid_pct=0,
                                               size=(256, 128), ds_tfms=ds_tfms, bs=cfg.SOLVER.IMS_PER_BATCH,
                                               val_bs=cfg.TEST.IMS_PER_BATCH)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        data_sampler = RandomIdentitySampler(train_names, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        data_bunch.train_dl = data_bunch.train_dl.new(shuffle=False, sampler=data_sampler)

    data_bunch.add_test(test_fnames)
    data_bunch.normalize(imagenet_stats)

    return data_bunch, test_labels, len(query_names)
