import sys
import unittest

import torch
from torch import nn

import sys
sys.path.append('.')
from modeling import *
from config import cfg

class MyTestCase(unittest.TestCase):
    def test_model(self):
        cfg.MODEL.WITH_IBN = True
        cfg.MODEL.PRETRAIN_PATH = '/home/user01/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar'
        net = build_model(cfg, 100)
        y = net(torch.randn(2, 3, 256, 128))
        from ipdb import set_trace; set_trace()
        # net1 = ResNet.from_name('resnet50', 1, True)
        # for i in net1.named_parameters():
            # print(i[0])
        # net2 = resnet50_ibn_a(1)
        # print('*'*10)
        # for i in net2.named_parameters():
            # print(i[0])


if __name__ == '__main__':
    unittest.main()
