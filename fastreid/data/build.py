# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
from torch.utils.data import DataLoader

from . import samplers
from .common import ReidDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms


def build_reid_train_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)

    logger = logging.getLogger(__name__)
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        logger.info('prepare training set {}'.format(d))
        dataset = DATASET_REGISTRY.get(d)()
        train_img_items.extend(dataset.train)

    train_set = ReidDataset(train_img_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    num_instance = cfg.DATALOADER.NUM_INSTANCE

    if cfg.DATALOADER.PK_SAMPLER:
        data_sampler = samplers.RandomIdentitySampler(train_set.img_items, batch_size, num_instance)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return train_loader


def build_reid_test_loader(cfg, dataset_name):
    # tng_tfms = build_transforms(cfg, is_train=True)
    test_transforms = build_transforms(cfg, is_train=False)

    logger = logging.getLogger(__name__)
    logger.info('prepare test set {}'.format(dataset_name))
    dataset = DATASET_REGISTRY.get(dataset_name)()
    query_names, gallery_names = dataset.query, dataset.gallery
    test_img_items = query_names + gallery_names

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH
    # train_img_items = list()
    # for d in cfg.DATASETS.NAMES:
    #     dataset = init_dataset(d)
    #     train_img_items.extend(dataset.train)

    # tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)

    # tng_set = ReidDataset(query_names + gallery_names, tng_tfms, False)
    # tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
    #                             num_workers=num_workers, collate_fn=fast_collate_fn, pin_memory=True, drop_last=True)
    test_set = ReidDataset(test_img_items, test_transforms, relabel=False)
    test_loader = DataLoader(test_set, batch_size, num_workers=num_workers,
                             collate_fn=trivial_batch_collator, pin_memory=True)
    return test_loader, len(query_names)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
