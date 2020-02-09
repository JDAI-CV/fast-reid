# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.modeling.backbones import *
from fastreid.modeling.backbones.resnet import Bottleneck
from fastreid.modeling.model_utils import *
from fastreid.modeling.heads import *
from fastreid.layers import BatchDrop


class BDNet(nn.Module):
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
        self.num_classes = num_classes
        if 'resnet' in backbone:
            self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
            self.base.load_pretrain(model_path)
            self.in_planes = 2048
        elif 'osnet' in backbone:
            if with_ibn:
                self.base = osnet_ibn_x1_0(pretrained=pretrain)
            else:
                self.base = osnet_x1_0(pretrained=pretrain)
            self.in_planes = 512
        else:
            print(f'not support {backbone} backbone')

        # global branch
        self.global_reduction = nn.Sequential(
            nn.Conv2d(self.in_planes, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_bn = bn2d_no_bias(512)
        self.global_classifier = nn.Linear(512, self.num_classes, bias=False)

        # mask brach
        self.part = Bottleneck(2048, 512)
        self.batch_drop = BatchDrop(1.0, 0.33)
        self.part_pool = nn.AdaptiveMaxPool2d(1)

        self.part_reduction = nn.Sequential(
            nn.Conv2d(self.in_planes, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        self.part_bn = bn2d_no_bias(1024)
        self.part_classifier = nn.Linear(1024, self.num_classes, bias=False)

        # initialize 
        self.part.apply(weights_init_kaiming)
        self.global_reduction.apply(weights_init_kaiming)
        self.part_reduction.apply(weights_init_kaiming)
        self.global_classifier.apply(weights_init_classifier)
        self.part_classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        # feature extractor
        feat = self.base(x)

        # global branch
        g_feat = self.global_reduction(feat)
        g_feat = self.gap(g_feat)  # (bs, 512, 1, 1)
        g_bn_feat = self.global_bn(g_feat)  # (bs, 512, 1, 1)
        g_bn_feat = g_bn_feat.view(-1, g_bn_feat.shape[1])  # (bs, 512)

        # mask branch
        p_feat = self.part(feat)
        p_feat = self.batch_drop(p_feat)
        p_feat = self.part_pool(p_feat)  # (bs, 512, 1, 1)
        p_feat = self.part_reduction(p_feat)
        p_bn_feat = self.part_bn(p_feat)
        p_bn_feat = p_bn_feat.view(-1, p_bn_feat.shape[1])  # (bs, 512)

        if self.training:
            global_cls = self.global_classifier(g_bn_feat)
            part_cls = self.part_classifier(p_bn_feat)
            return global_cls, part_cls, g_feat.view(-1, g_feat.shape[1]), p_feat.view(-1, p_feat.shape[1])

        return torch.cat([g_bn_feat, p_bn_feat], dim=1)

    def load_params_wo_fc(self, state_dict):
        state_dict.pop('global_classifier.weight')
        state_dict.pop('part_classifier.weight')

        res = self.load_state_dict(state_dict, strict=False)
        print(f'missing keys {res.missing_keys}')
        # assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    def unfreeze_all_layers(self,):
        self.train()
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_specific_layer(self, names):
        if isinstance(names, str):
            names = [names]
        
        for name, module in self.named_children():
            if name in names:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
