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
from modeling.backbones import *
from modeling.mgn import MGN
from modeling.mgn_plus import MGN_P

cfg.MODEL.BACKBONE = 'resnet50'
cfg.MODEL.WITH_IBN = False
# cfg.MODEL.PRETRAIN_PATH = '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar'

net = MGN_P('resnet50', 100, 1, False, None, cfg.MODEL.STAGE_WITH_GCB, cfg.MODEL.PRETRAIN, cfg.MODEL.PRETRAIN_PATH)
# net = MGN('resnet50', 100, 2, False,None, cfg.MODEL.STAGE_WITH_GCB, cfg.MODEL.PRETRAIN, cfg.MODEL.PRETRAIN_PATH)
# net.eval()
# net = net.cuda()
x = torch.randn(10, 3, 256, 128)
y = net(x)
from ipdb import set_trace; set_trace()
# label = torch.ones(10).long().cuda()
# y = net(x, label)


