# -*- coding: utf-8 -*-

import os
import math

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image

from fastreid.data.data_utils import read_image

__all__ = ['augment_pos_image', 'augment_neg_image']

pos_augmenter = iaa.Sequential(
        [
            iaa.CropAndPad(percent=(-0.15, 0.15), pad_mode=ia.ALL),
            iaa.Rotate(rotate=(-15, 15), mode=ia.ALL)
        ])


def augment_pos_image(img: Image.Image) -> Image.Image:
    img = self.pos_ia_augmenter.augment_image(np.array(img))
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def augment_neg_image(img_root: str, neg_list: list[str], img: Image.Image) -> Image.Image:
    img = img.copy()
    img_w, img_h = img.size
    alpha = random.uniform(0.3, 0.7)
    min_bound = 0.2
    max_bound = 0.3
    num_patches = [2, 3, 4, 5]
    new_img = np.array(img).astype(np.float32)
    for _ in range(random.choice(num_patches)):
        ref_img_path = os.path.join(img_root, random.choice(neg_list))
        ref_img = Image.open(ref_img_path)


        # 计算要crop的相对位置
        crop_w, crop_h = random.uniform(min_bound, max_bound), random.uniform(min_bound, max_bound)
        # 0.125 = 0.25 / 2, 0.875 = 1 - 0.125, 确保中心加减宽高在图像内部
        crop_center_x, crop_center_y = np.clip(np.random.normal(loc=0.5, scale=0.35 / 2, size=(2,)), 
                a_min=max_bound / 2, a_max=1 - max_bound / 2)
        crop_xmin, crop_ymin = crop_center_x - crop_w / 2, crop_center_y - crop_h / 2
        crop_xmax, crop_ymax = crop_xmin + crop_w, crop_ymin + crop_h
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = np.clip([crop_xmin, crop_ymin, crop_xmax, crop_ymax], a_min=0.0, a_max=1.0)
        crop_w, crop_h = crop_xmax - crop_xmin, crop_ymax - crop_ymin
        # 图1上要被粘贴的位置
        paste_xmin, paste_ymin = math.floor(crop_xmin * img_w), math.floor(crop_ymin * img_h)
        paste_xmax, paste_ymax = math.floor(crop_xmax * img_w), math.floor(crop_ymax * img_h)

        # 生成mask
        ref_img = ref_img.resize((img_w, img_h))
        mask = np.zeros_like(new_img) 
        mask[paste_ymin: paste_ymax, paste_xmin: paste_xmax, :] = 1

        new_img = new_img * (1 - mask) + ref_img * mask * alpha + new_img * mask * (1 - alpha)
    new_img = Image.fromarray(new_img.astype('uint8')).convert('RGB')
    return new_img
