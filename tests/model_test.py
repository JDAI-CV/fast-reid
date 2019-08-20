import sys
import unittest

import torch
from torch import nn

import sys
sys.path.append('.')
from modeling.backbones import *
from config import cfg


class MyTestCase(unittest.TestCase):
    def test_model(self):
        net1 = ResNet.from_name('resnet50', 1, True)
        for i in net1.named_parameters():
            print(i[0])
        net2 = resnet50_ibn_a(1)
        # print('*'*10)
        # for i in net2.named_parameters():
        #     print(i[0])


if __name__ == '__main__':
    unittest.main()
