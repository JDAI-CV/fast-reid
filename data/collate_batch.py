# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch


def tng_collate_fn(batch):
    imgs, pids, camids = zip(*batch)
    return torch.stack(imgs, dim=0), torch.tensor(pids).long()

