# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import re

from fastai.vision import *

from .datasets import CUHK03
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def get_data_bunch(cfg):
    ds_tfms = build_transforms(cfg)

    def _process_dir(dir_path):
        img_paths = []
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d*)')
        v_paths = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            pid = int(pid)
            if pid == -1: continue  # junk images are just ignored
            v_paths.append([img_path,pid,camid])
        return v_paths

    market_train_path = 'datasets/Market-1501-v15.09.15/bounding_box_train'
    duke_train_path = 'datasets/DukeMTMC-reID/bounding_box_train'
    cuhk03_train_path = 'datasets/cuhk03/'

    market_query_path = 'datasets/Market-1501-v15.09.15/query'
    marker_gallery_path = 'datasets/Market-1501-v15.09.15/bounding_box_test'
    duke_query_path = 'datasets/DukeMTMC-reID/query'
    duke_gallery_path = 'datasets/DukeMTMC-reID/bounding_box_test'

    train_img_names = list()
    for d in cfg.DATASETS.NAMES:
        if d == 'market1501':   train_img_names.extend(_process_dir(market_train_path))
        elif d == 'duke':       train_img_names.extend(_process_dir(duke_train_path))
        elif d == 'cuhk03':     train_img_names.extend(CUHK03().train)
        else:                   raise NameError(f'{d} is not available')
        
    train_names = [i[0] for i in train_img_names]

    if cfg.DATASETS.TEST_NAMES == "market1501":
        query_names = _process_dir(market_query_path)
        gallery_names = _process_dir(marker_gallery_path)
    elif cfg.DATASETS.TEST_NAMES == 'duke':
        query_names = _process_dir(duke_query_path)
        gallery_names = _process_dir(duke_gallery_path)
    else:
        print(f"not support {cfg.DATASETS.TEST_NAMES} test set")

    test_fnames = []
    test_labels = []
    for i in query_names+gallery_names:
        test_fnames.append(i[0])
        test_labels.append(i[1:])

    def get_labels(file_path):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:
            prefix = 'cuhk'
            pid = '_'.join(file_path.split('/')[-1].split('_')[0:2])
        else:
            prefix = file_path.split('/')[1]
            pat = re.compile(r'([-\d]+)_c(\d)')
            pid, _ = pat.search(file_path).groups()
        return prefix + '_' + pid

    data_bunch = ImageDataBunch.from_name_func('datasets', train_names, label_func=get_labels, valid_pct=0,
                                               size=cfg.INPUT.SIZE_TRAIN, ds_tfms=ds_tfms, bs=cfg.SOLVER.IMS_PER_BATCH,
                                               val_bs=cfg.TEST.IMS_PER_BATCH)

    if cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler(train_names, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        data_bunch.train_dl = data_bunch.train_dl.new(shuffle=False, sampler=data_sampler)

    data_bunch.add_test(test_fnames)
    data_bunch.normalize(imagenet_stats)

    return data_bunch, test_labels, len(query_names)
