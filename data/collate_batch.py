# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np

def fast_collate_fn(batch):
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
    return tensor, torch.tensor(pids).long(), camids


# def dcl_collate_fn(batch):
#     imgs, swap_imgs, pids = zip(*batch)
#     imgs = torch.stack(imgs, dim=0)
#     swap_imgs = torch.stack(swap_imgs, dim=0)
#     # pids *= 2
#     swap_labels = [1] * imgs.size()[0] + [0] * swap_imgs.size()[0]
#     # return torch.cat([imgs, swap_imgs], dim=0), (tensor(pids).long(), tensor(swap_labels).long())
#     return imgs, (tensor(pids).long(), tensor(swap_labels).long())
