# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
from torch.utils.data import DataLoader

from .common import ReidDataset
from .datasets import init_dataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def build_reid_train_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)

    print('prepare training set ...')
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = init_dataset(d)
        train_img_items.extend(dataset.train)
    # for d in ['market1501', 'dukemtmc', 'msmt17']:
    #     dataset = init_dataset(d, combineall=True)
    #     train_img_items.extend(dataset.train)

    train_set = ReidDataset(train_img_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # num_workers = 0
    data_sampler = None
    if cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler(train_set.img_items, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)

    train_loader = DataLoader(train_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=(data_sampler is None),
                              num_workers=num_workers, sampler=data_sampler, collate_fn=trivial_batch_collator,
                              pin_memory=True, drop_last=True)

    #
    # test_set = ReidDataset(test_img_items, test_transforms, relabel=False)
    # test_dataloader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
    #                              pin_memory=True)
    # return tng_dataloader, test_dataloader, tng_set.c, len(query_names)
    return train_loader


def build_reid_test_loader(cfg):
    # tng_tfms = build_transforms(cfg, is_train=True)
    test_transforms = build_transforms(cfg, is_train=False)

    print('prepare test set ...')
    dataset = init_dataset(cfg.DATASETS.TEST[0])
    query_names, gallery_names = dataset.query, dataset.gallery
    test_img_items = list(set(query_names) | set(gallery_names))

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # train_img_items = list()
    # for d in cfg.DATASETS.NAMES:
    #     dataset = init_dataset(d)
    #     train_img_items.extend(dataset.train)

    # tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)

    # tng_set = ReidDataset(query_names + gallery_names, tng_tfms, False)
    # tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
    #                             num_workers=num_workers, collate_fn=fast_collate_fn, pin_memory=True, drop_last=True)
    test_set = ReidDataset(test_img_items, test_transforms, relabel=False)
    test_loader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
                             collate_fn=trivial_batch_collator, pin_memory=True)
    return test_loader, len(query_names)
    # return tng_dataloader, test_dataloader, len(query_names)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
