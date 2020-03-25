import unittest

import torch

import sys
sys.path.append('.')
from fastreid.config import cfg
from fastreid.modeling.backbones import build_resnet_backbone
from fastreid.modeling.backbones.resnet_ibn_a import se_resnet101_ibn_a
from torch import nn


class MyTestCase(unittest.TestCase):
    def test_se_resnet101(self):
        cfg.MODEL.BACKBONE.NAME = 'resnet101'
        cfg.MODEL.BACKBONE.DEPTH = 101
        cfg.MODEL.BACKBONE.WITH_IBN = True
        cfg.MODEL.BACKBONE.WITH_SE = True
        cfg.MODEL.BACKBONE.PRETRAIN_PATH = '/export/home/lxy/.cache/torch/checkpoints/se_resnet101_ibn_a.pth.tar'

        net1 = build_resnet_backbone(cfg)
        net1.cuda()
        net2 = nn.DataParallel(se_resnet101_ibn_a())
        res = net2.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAIN_PATH)['state_dict'], strict=False)
        net2.cuda()
        x = torch.randn(10, 3, 256, 128).cuda()
        y1 = net1(x)
        y2 = net2(x)
        assert y1.sum() == y2.sum(), 'train mode problem'
        net1.eval()
        net2.eval()
        y1 = net1(x)
        y2 = net2(x)
        assert y1.sum() == y2.sum(), 'eval mode problem'


if __name__ == '__main__':
    unittest.main()
