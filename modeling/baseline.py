# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .backbones import *
from .losses.cosface import AddMarginProduct


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

    def __init__(self, 
                 backbone, 
                 num_classes, 
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb, 
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        try:    self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except: print(f'not support {backbone} backbone')

        if pretrain: self.base.load_pretrain(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier = AddMarginProduct(self.in_planes, self.num_classes, s=30, m=0.3)

        self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(-1, global_feat.size()[1])
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(feat, label)  # (2*b, class)
            # adv_score = self.classifier_swap(feat)  # (2*b, 2)
            # return cls_score, adv_score, global_feat  # global feature for triplet loss
            return cls_score, global_feat
        else:
            return feat

    def load_params_wo_fc(self, state_dict):
        state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'