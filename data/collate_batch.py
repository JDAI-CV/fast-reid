# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
import cv2


def fast_collate_fn(batch):
    img_data, pids, camids = zip(*batch)
    has_mask = False
    if isinstance(img_data[0], tuple):
        has_mask = True
        imgs, masks = zip(*img_data)
    else:
        imgs = img_data
    is_ndarray = isinstance(imgs[0], np.ndarray)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        tensor[i] += torch.from_numpy(numpy_array)
    if has_mask:
        mask_tensor = torch.stack(masks, dim=0)
        return tensor, mask_tensor, torch.tensor(pids).long(), camids
    else:
        return tensor, torch.tensor(pids).long(), camids


def test_collate_fn(batch):
    imgs, pids, camids = zip(*batch)
    is_ndarray = isinstance(imgs[0], np.ndarray)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        tensor[i] += torch.from_numpy(numpy_array)
    return tensor, pids, camids

