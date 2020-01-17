# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch
import random
import re

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

__all__ = ['ImageDataset']


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            from ipdb import set_trace; set_trace()
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, mask_transforms=None, relabel=True, query_len=0, return_mask=False):
        self.tfms, self.mask_tfms, self.relabel, self.query_len = transform, mask_transforms, relabel, query_len
        self.return_mask = return_mask

        self.pid2label = None
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                if self.return_mask:
                    pid = self.get_pids(item[0][0], item[1])  # path
                else:
                    pid = self.get_pids(item[0], item[1])
                self.img_items.append((item[0], pid, item[2]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid2label = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items

    @property
    def c(self):
        return len(self.pid2label) if self.pid2label is not None else 0

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if self.return_mask:
            # (img_path, pose_path), pid, camid = self.img_items[index]
            # pose_img = np.load(pose_path)
            # pose_img = pose_img.reshape(24, 8)
            # pose = Image.fromarray(pose_img)
            (img_path, mask_path), pid, camid = self.img_items[index]
            mask_img = np.array(Image.open(mask_path).convert('P'))
            mask_img[mask_img != 0] = 255
            mask = Image.fromarray(mask_img)
        else:
            img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        # mask = read_image(mask_path)
        # if index < self.query_len:
        #     w, h = img.size
        #     img = img.crop((0, 0, w, int(0.5*h)))

        # w, h = img.size
        # if w / h < 0.5:
        #     img = img
        # elif w / h < 1.5:
        #     new_h = int(128 * h / w)
        #     img = T.Resize((new_h, 128))(img)
        #     padding_h = 256 - new_h
        #     img = T.Pad(padding=((0, 0, 0, padding_h)))(img)
        # else:
        #     # print(f'not good image {index}')
        #     img = img

            # img = T.Resize((128, 128))(img)
            # new_image = Image.new("RGB", (w, h))
            # new_image.paste(img, (0, 0, w, int(0.5*h)))
            # img = new_image

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img transforms
        if self.tfms is not None:
            img = self.tfms(img)

        if self.return_mask:
            random.seed(seed)  # apply this seed to mask transforms
            if self.mask_tfms is not None:
                mask = self.mask_tfms(mask)
                # pose = self.mask_tfms(pose)
            # mask = T.ToTensor()(mask)
            # mask = mask.view(-1)
            mask = T.ToTensor()(mask)
            mask1 = F.avg_pool2d(mask, kernel_size=16, stride=16).view(-1)  # (192)
            mask2 = F.avg_pool2d(mask, kernel_size=32, stride=32).view(-1)  # (48)
            mask3 = F.avg_pool2d(mask, kernel_size=64, stride=32).view(-1)  # (33)
            mask_score = torch.cat([mask1, mask2, mask3], dim=0)  # (273)

        if self.relabel:
            pid = self.pid2label[pid]
        if self.return_mask:
            return (img, mask_score), pid, camid
        else:
            return img, pid, camid

    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:
            prefix = 'cuhk'
        else:
            prefix = file_path.split('/')[1]
        return prefix + '_' + str(pid)

