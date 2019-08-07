# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
from fastai.vision import *
sys.path.append('.')
from data import get_data_bunch
from config import cfg


if __name__ == '__main__':
    # cfg.INPUT.SIZE_TRAIN = (384, 128)
    data, label, num_q = get_data_bunch(cfg)
    # def get_ex(): return open_image('datasets/beijingStation/query/000245_c10s2_1561732033722.000000.jpg')
    # im = get_ex()
    print(data.train_ds[0])
    print(data.test_ds[0])
    from ipdb import set_trace; set_trace()
    # im.apply_tfms(crop_pad(size=(300, 300)))
