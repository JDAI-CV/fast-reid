# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn.functional as F
from torch import nn

from .resnet import ResNet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, model_path='/DATA/model_zoo/resnet50-19c8e357.pth'):
        super().__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(True)
        )
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        feat = self.base(x)
        feat = self.bottleneck(feat)
        global_feat = F.avg_pool2d(feat, feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            cls_score = self.classifier(global_feat)
            return cls_score, global_feat
        else:
            return global_feat

    def optim_policy(self):
        base_param_group = self.base.parameters()
        clf_param_group = self.classifier.parameters()
        return [
            {'params': base_param_group, 'lr_multi': 0.1},
            {'params': clf_param_group}
        ]


