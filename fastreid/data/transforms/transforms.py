# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

__all__ = ['RandomErasing', 'Cutout', 'random_angle_rotate', 'do_color', 'random_shift', 'random_scale']

import math
import random
from PIL import Image
import cv2

import numpy as np

from .functional import *


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=255 * (0.49735, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        img = np.asarray(img, dtype=np.uint8).copy()
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return Image.fromarray(img)
        return Image.fromarray(img)


class Cutout(object):
    def __init__(self, probability=0.5, size=64, mean=255 * [0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.size = size

    def __call__(self, img):
        img = np.asarray(img, dtype=np.uint8).copy()
        if random.uniform(0, 1) > self.probability:
            return img

        h = self.size
        w = self.size
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return img
        return img


class random_angle_rotate(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def __call__(self, image, angles=[-30, 30]):
        image = np.asarray(image, dtype=np.uint8).copy()
        if random.uniform(0, 1) > self.probability:
            return image

        angle = random.randint(0, angles[1] - angles[0]) + angles[0]
        image = self.rotate(image, angle)
        return image


class do_color(object):
    """docstring for do_color"""

    def __init__(self, probability=0.5):
        self.probability = probability

    def do_brightness_shift(self, image, alpha=0.125):
        image = image.astype(np.float32)
        image = image + alpha * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def do_brightness_multiply(self, image, alpha=1):
        image = image.astype(np.float32)
        image = alpha * image
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def do_contrast(self, image, alpha=1.0):
        image = image.astype(np.float32)
        gray = image * np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha * image + gray
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    def do_gamma(self, image, gamma=1.0):

        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)  # apply gamma correction using the lookup table

    def do_clahe(self, image, clip=2, grid=16):
        grid = int(grid)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray, a, b = cv2.split(lab)
        gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
        lab = cv2.merge((gray, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image

        index = random.randint(0, 4)
        if index == 0:
            image = self.do_brightness_shift(image, 0.1)
        elif index == 1:
            image = self.do_gamma(image, 1)
        elif index == 2:
            image = self.do_clahe(image)
        elif index == 3:
            image = self.do_brightness_multiply(image)
        elif index == 4:
            image = self.do_contrast(image)
        return image


class random_shift(object):
    """docstring for do_color"""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image

        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        w = random.randint(0, 20) - 10
        h = random.randint(0, 30) - 15
        zero_image[max(0, w): min(w + width, width), max(h, 0): min(h + height, height)] = \
            image[max(0, -w): min(-w + width, width), max(-h, 0): min(-h + height, height)]
        image = zero_image.copy()
        return image


class random_scale(object):
    """docstring for do_color"""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image

        scale = random.random() * 0.1 + 0.9
        assert 0.9 <= scale <= 1
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = cv2.resize(image, (new_height, new_width))
        start_w = random.randint(0, width - new_width)
        start_h = random.randint(0, height - new_height)
        zero_image[start_w: start_w + new_width,
        start_h:start_h + new_height] = image
        image = zero_image.copy()
        return image
