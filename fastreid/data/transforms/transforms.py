# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

__all__ = ['ToTensor', 'RandomErasing', 'RandomPatch', 'AugMix', ]

import math
import random
from collections import deque

import numpy as np
from PIL import Image

from .functional import to_tensor, augmentations_reid


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


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
        img = np.asarray(img, dtype=np.float32).copy()
        if random.uniform(0, 1) > self.probability:
            return img

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
                return img
        return img


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    Args:
        aug_prob_coeff: Probability distribution coefficients.
        mixture_width: Number of augmentation chains to mix per augmented example.
        mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
        severity: Severity of underlying augmentation operators (between 1 to 10).
    """

    def __init__(self, aug_prob_coeff=1, mixture_width=3, mixture_depth=-1, severity=1):
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.severity = severity
        self.aug_list = augmentations_reid

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.
        Returns:
          mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        image = np.asarray(image, dtype=np.float32).copy()
        mix = np.zeros_like(image)
        h, w = image.shape[0], image.shape[1]
        for i in range(self.mixture_width):
            image_aug = Image.fromarray(image.copy().astype(np.uint8))
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.severity, (w, h))
            mix += ws[i] * np.asarray(image_aug, dtype=np.float32)

        mixed = (1 - m) * image + m * mix
        return mixed

# class ColorJitter(object):
#     """docstring for do_color"""
#
#     def __init__(self, probability=0.5):
#         self.probability = probability
#
#     def do_brightness_shift(self, image, alpha=0.125):
#         image = image.astype(np.float32)
#         image = image + alpha * 255
#         image = np.clip(image, 0, 255).astype(np.uint8)
#         return image
#
#     def do_brightness_multiply(self, image, alpha=1):
#         image = image.astype(np.float32)
#         image = alpha * image
#         image = np.clip(image, 0, 255).astype(np.uint8)
#         return image
#
#     def do_contrast(self, image, alpha=1.0):
#         image = image.astype(np.float32)
#         gray = image * np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
#         gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
#         image = alpha * image + gray
#         image = np.clip(image, 0, 255).astype(np.uint8)
#         return image
#
#     # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
#     def do_gamma(self, image, gamma=1.0):
#         table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
#                           for i in np.arange(0, 256)]).astype("uint8")
#
#         return cv2.LUT(image, table)  # apply gamma correction using the lookup table
#
#     def do_clahe(self, image, clip=2, grid=16):
#         grid = int(grid)
#
#         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         gray, a, b = cv2.split(lab)
#         gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
#         lab = cv2.merge((gray, a, b))
#         image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
#         return image
#
#     def __call__(self, image):
#         if random.uniform(0, 1) > self.probability:
#             return image
#
#         image = np.asarray(image, dtype=np.uint8).copy()
#         index = random.randint(0, 4)
#         if index == 0:
#             image = self.do_brightness_shift(image, 0.1)
#         elif index == 1:
#             image = self.do_gamma(image, 1)
#         elif index == 2:
#             image = self.do_clahe(image)
#         elif index == 3:
#             image = self.do_brightness_multiply(image)
#         elif index == 4:
#             image = self.do_contrast(image)
#         return image


# class random_shift(object):
#     """docstring for do_color"""
#
#     def __init__(self, probability=0.5):
#         self.probability = probability
#
#     def __call__(self, image):
#         if random.uniform(0, 1) > self.probability:
#             return image
#
#         width, height, d = image.shape
#         zero_image = np.zeros_like(image)
#         w = random.randint(0, 20) - 10
#         h = random.randint(0, 30) - 15
#         zero_image[max(0, w): min(w + width, width), max(h, 0): min(h + height, height)] = \
#             image[max(0, -w): min(-w + width, width), max(-h, 0): min(-h + height, height)]
#         image = zero_image.copy()
#         return image
#
#
# class random_scale(object):
#     """docstring for do_color"""
#
#     def __init__(self, probability=0.5):
#         self.probability = probability
#
#     def __call__(self, image):
#         if random.uniform(0, 1) > self.probability:
#             return image
#
#         scale = random.random() * 0.1 + 0.9
#         assert 0.9 <= scale <= 1
#         width, height, d = image.shape
#         zero_image = np.zeros_like(image)
#         new_width = round(width * scale)
#         new_height = round(height * scale)
#         image = cv2.resize(image, (new_height, new_width))
#         start_w = random.randint(0, width - new_width)
#         start_h = random.randint(0, height - new_height)
#         zero_image[start_w: start_w + new_width,
#         start_h:start_h + new_height] = image
#         image = zero_image.copy()
#         return image
