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
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.utils import comm
from .attr_dataset import AttrDataset

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_attr_train_loader(cfg):
    cfg = cfg.clone()
    cfg.defrost()

    train_items = list()
    attr_dict = None
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        if attr_dict is not None:
            assert attr_dict == dataset.attr_dict, "attr_dict in {} does not match with previous ones".format(d)
        else:
            attr_dict = dataset.attr_dict
        train_items.extend(dataset.train)

    train_transforms = build_transforms(cfg, is_train=True)
    train_set = AttrDataset(train_items, attr_dict, train_transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

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


def build_attr_test_loader(cfg, dataset_name):
    cfg = cfg.clone()
    cfg.defrost()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.test

    test_transforms = build_transforms(cfg, is_train=False)
    test_set = AttrDataset(test_items, dataset.attr_dict, test_transforms)

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
