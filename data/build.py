# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import re

from torch.utils.data import DataLoader

from .collate_batch import fast_collate_fn
from .datasets import ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms
from .datasets import init_dataset


def get_dataloader(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    val_tfms = build_transforms(cfg, is_train=False)

    print('prepare training set ...')
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        # dataset = init_dataset(d, combineall=True)
        dataset = init_dataset(d)
        train_img_items.extend(dataset.train)

    print('prepare test set ...')
    dataset = init_dataset(cfg.DATASETS.TEST_NAMES)
    query_names, gallery_names = dataset.query, dataset.gallery

    tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_sampler = None
    if cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler(tng_set.img_items, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)

    tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=(data_sampler is None),
                                num_workers=num_workers, sampler=data_sampler,
                                collate_fn=fast_collate_fn, pin_memory=True)

    val_set = ImageDataset(query_names+gallery_names, val_tfms, relabel=False)
    val_dataloader = DataLoader(val_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers, 
                                collate_fn=fast_collate_fn, pin_memory=True)
    return tng_dataloader, val_dataloader, tng_set.c, len(query_names)


def get_test_dataloader(cfg):
    val_tfms = build_transforms(cfg, is_train=False)

    print('prepare test set ...')
    dataset = init_dataset(cfg.DATASETS.TEST_NAMES)
    query_names, gallery_names = dataset.query, dataset.gallery

    num_workers = cfg.DATALOADER.NUM_WORKERS

    test_set = ImageDataset(query_names+gallery_names, val_tfms, relabel=False)
    test_dataloader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
                                collate_fn=fast_collate_fn, pin_memory=True)
    return test_dataloader, len(query_names)
