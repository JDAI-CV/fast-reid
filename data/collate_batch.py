# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from fastai.vision import *


def test_collate_fn(batch):
    imgs, label = zip(*batch)
    imgs = to_data(imgs)
    pids = []
    camids = []
    for i in label:
        pids.append(i.obj[0])
        camids.append(i.obj[1])
    return torch.stack(imgs, dim=0), (torch.LongTensor(pids), torch.LongTensor(camids))
