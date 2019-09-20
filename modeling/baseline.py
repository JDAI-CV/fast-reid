# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .backbones import *
from .losses.cosface import AddMarginProduct
from .utils import *


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
        try:
            self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier = AddMarginProduct(self.in_planes, self.num_classes, s=20, m=0.3)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(-1, global_feat.size()[1])
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(feat)
            # cls_score = self.classifier(feat, label)
            return cls_score, global_feat
        else:
            return feat

    def load_params_wo_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict
        state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'
