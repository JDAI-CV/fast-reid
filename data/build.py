# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import re

from torch.utils.data import DataLoader

from .collate_batch import tng_collate_fn
from .datasets import ImageDataset, CUHK03
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def get_dataloader(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    val_tfms = build_transforms(cfg, is_train=False)

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
    duek_gallery_path = 'datasets/DukeMTMC-reID/bounding_box_test'

    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        if d == 'market1501':   train_img_items.extend(_process_dir(market_train_path))
        elif d == 'duke':       train_img_items.extend(_process_dir(duke_train_path))
        elif d == 'cuhk03':     train_img_items.extend(CUHK03().train)
        else:
            raise NameError(f"{d} is not available")

    if cfg.DATASETS.TEST_NAMES == "market1501":
        query_names = _process_dir(market_query_path)
        gallery_names = _process_dir(marker_gallery_path)
    elif cfg.DATASETS.TEST_NAMES == 'duke':
        query_names = _process_dir(duke_query_path)
        gallery_names = _process_dir(duek_gallery_path)
    else:
        print(f"not support {cfg.DATASETS.TEST_NAMES} test set")

    num_workers = min(16, len(os.sched_getaffinity(0)))

    tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
                                    num_workers=num_workers, collate_fn=tng_collate_fn,
                                    pin_memory=True)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler(train_img_items, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, sampler=data_sampler,
                                    num_workers=num_workers, collate_fn=tng_collate_fn,
                                    pin_memory=True)
    else:
        raise NameError(f"{cfg.DATALOADER.SAMPLER} sampler is not support")

    val_set = ImageDataset(query_names+gallery_names, val_tfms, relabel=False)
    val_dataloader = DataLoader(val_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_cpus)
    return tng_dataloader, val_dataloader, tng_set.c, len(query_names)
