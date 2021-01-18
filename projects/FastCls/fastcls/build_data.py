# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os

import torch
from torch.utils.data import DataLoader

from fastreid.data import samplers
from fastreid.data.build import fast_batch_collator
from fastreid.data.common import CommDataset
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils import comm
from .build_transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_cls_train_loader(cfg, mapper=None, **kwargs):
    cfg = cfg.clone()

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, **kwargs)
        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=True)

    train_set = CommDataset(train_items, transforms, relabel=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if cfg.DATALOADER.PK_SAMPLER:
        if cfg.DATALOADER.NAIVE_WAY:
            data_sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
        else:
            data_sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def build_cls_test_loader(cfg, dataset_name, mapper=None, **kwargs):
    cfg = cfg.clone()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query

    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=False)

    test_set = CommDataset(test_items, transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=4,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader
