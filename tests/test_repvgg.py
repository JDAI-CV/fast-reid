import sys
import unittest

import torch

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.modeling.backbones import build_backbone


class MyTestCase(unittest.TestCase):
    def test_fusebn(self):
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.BACKBONE.NAME = 'build_repvgg_backbone'
        cfg.MODEL.BACKBONE.DEPTH = 'B1g2'
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = build_backbone(cfg)
        model.eval()

        test_inp = torch.randn((1, 3, 256, 128))

        y = model(test_inp)

        model.deploy(mode=True)
        from ipdb import set_trace; set_trace()
        fused_y = model(test_inp)

        print("final error :", torch.max(torch.abs(fused_y - y)).item())


if __name__ == '__main__':
    unittest.main()
