# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import random
from PIL import Image

__all__ = ['swap']


def swap(img, crop):
    def crop_image(image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    widthcut, highcut = img.size
    img = img.crop((10, 10, widthcut - 10, highcut - 10))
    images = crop_image(img, crop)
    pro = 5
    if pro >= 5:
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 1
        RAN = 2
        for i in range(crop[1] * crop[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
            if count_x == crop[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                random.shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
        random_im = []
        for line in tmpy:
            random_im.extend(line)

        # random.shuffle(images)
        width, high = img.size
        iw = int(width / crop[0])
        ih = int(high / crop[1])
        toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
        x = 0
        y = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            x += 1
            if x == crop[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    return toImage
