# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch

import sys
sys.path.append(".")
from config import cfg
from modeling import build_model
from modeling.bdnet import BDNet

cfg.MODEL.BACKBONE = 'resnet50'
cfg.MODEL.WITH_IBN = False
# cfg.MODEL.PRETRAIN_PATH = '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar'

net = BDNet('resnet50', 100, 1, False, None, cfg.MODEL.STAGE_WITH_GCB, False)
y = net(torch.randn(2, 3, 256, 128))
print(3)
# net = MGN_P('resnet50', 100, 1, False, None, cfg.MODEL.STAGE_WITH_GCB, cfg.MODEL.PRETRAIN, cfg.MODEL.PRETRAIN_PATH)
# net = MGN('resnet50', 100, 2, False,None, cfg.MODEL.STAGE_WITH_GCB, cfg.MODEL.PRETRAIN, cfg.MODEL.PRETRAIN_PATH)
# net.eval()
# net = net.cuda()
# x = torch.randn(10, 3, 256, 128)
# y = net(x)
# net = osnet_x1_0(False)
# net(torch.randn(1, 3, 256, 128))
# from ipdb import set_trace; set_trace()
# label = torch.ones(10).long().cuda()
# y = net(x, label)


