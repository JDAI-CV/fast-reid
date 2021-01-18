# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import DataLoader

from fastreid.data import samplers
from fastreid.data.build import fast_batch_collator, _root
from fastreid.data.common import CommDataset
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils import comm


class FaceCommDataset(CommDataset):
    def __init__(self, img_items, labels):
        self.img_items = img_items
        self.labels = labels

    def __getitem__(self, index):
        img = torch.tensor(self.img_items[index]) * 127.5 + 127.5
        return {
            "images": img,
        }


def build_face_test_loader(cfg, dataset_name, **kwargs):
    cfg = cfg.clone()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    if comm.is_main_process():
        dataset.show_test()

    test_set = FaceCommDataset(dataset.carray, dataset.is_same)

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
    return test_loader, test_set.labels

