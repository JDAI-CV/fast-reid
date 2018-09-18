# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn

from .resnet import ResNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=10, last_stride=1, model_path='/home/test2/.torch/models/resnet50-19c8e357.pth'):
        super().__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # nn.Linear(self.in_planes, 512),
        # nn.BatchNorm1d(512),
        # nn.LeakyReLU(0.1),
        # nn.Dropout(p=0.5)
        # )
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat
        else:
            return feat


if __name__ == '__main__':
    net = Baseline()
    import torch

    x = torch.ones(2, 3, 256, 128)
    y = net(x)
    from IPython import embed

    embed()
