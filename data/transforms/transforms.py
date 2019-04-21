# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

from fastai.vision import *
from fastai.vision.image import *


def _random_erasing(x, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                    mean=(np.array(imagenet_stats[1]) + 1) * imagenet_stats[0]):
    if random.uniform(0, 1) > probability:
        return x

    for attempt in range(100):
        area = x.size()[1] * x.size()[2]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < x.size()[2] and h < x.size()[1]:
            x1 = random.randint(0, x.size()[1] - h)
            y1 = random.randint(0, x.size()[2] - w)
            if x.size()[0] == 3:
                x[0, x1:x1 + h, y1:y1 + w] = mean[0]
                x[1, x1:x1 + h, y1:y1 + w] = mean[1]
                x[2, x1:x1 + h, y1:y1 + w] = mean[2]
            else:
                x[0, x1:x1 + h, y1:y1 + w] = mean[0]
            return x


RandomErasing = TfmPixel(_random_erasing)
