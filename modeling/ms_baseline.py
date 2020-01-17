# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import *
from .utils import *
from .losses import *
from layers import bn_no_bias, GeM


class ClassBlock(nn.Module):
    """
    Define the bottleneck and classifier layer
    |--bn--|--relu--|--linear--|--classifier--|
    """
    def __init__(self, in_features, num_classes, relu=True, num_bottleneck=512):
        super().__init__()
        block1 = []
        block1 += [nn.BatchNorm1d(in_features)]
        if relu:
            block1 += [nn.LeakyReLU(0.1)]
        block1 += [nn.Linear(in_features, num_bottleneck, bias=False)]
        self.block1 = nn.Sequential(*block1)

        self.bnneck = bn_no_bias(num_bottleneck)

        # self.classifier = nn.Linear(num_bottleneck, num_classes, bias=False)
        self.classifier = CircleLoss(num_bottleneck, num_classes, s=256, m=0.25)

    def init_parameters(self):
        self.block1.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        x = self.block1(x)
        x = self.bnneck(x)
        if self.training:
            cls_out = self.classifier(x, label)
            return cls_out
        else:
            return x


class MSBaseline(nn.Module):
    def __init__(self,
                 backbone,
                 num_classes,
                 last_stride,
                 with_ibn=False,
                 with_se=False,
                 gcb=None,
                 stage_with_gcb=[False, False, False, False],
                 pretrain=True,
                 model_path=''):
        super().__init__()
        if 'resnet' in backbone:
            self.base = ResNet.from_name(backbone, pretrain, last_stride, with_ibn, with_se, gcb,
                                         stage_with_gcb, model_path=model_path)
            self.in_planes = 2048
        elif 'osnet' in backbone:
            if with_ibn:
                self.base = osnet_ibn_x1_0(pretrained=pretrain)
            else:
                self.base = osnet_x1_0(pretrained=pretrain)
            self.in_planes = 512
        else:
            print(f'not support {backbone} backbone')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.gap = GeM()

        self.num_classes = num_classes

        self.classifier1 = ClassBlock(in_features=1024, num_classes=num_classes)
        self.classifier2 = ClassBlock(in_features=2048, num_classes=num_classes)

    def forward(self, x, label=None, **kwargs):
        x4, x3 = self.base(x)  # (bs, 2048, 16, 8)
        x3_max = self.maxpool(x3)
        x3_max = x3_max.view(x3_max.shape[0], -1)  # (bs, 2048)
        x3_avg = self.avgpool(x3)
        x3_avg = x3_avg.view(x3_avg.shape[0], -1)  # (bs, 2048)
        x3_feat = x3_max + x3_avg
        # x3_feat = self.gap(x3)  # (bs, 2048, 1, 1)
        # x3_feat = x3_feat.view(x3_feat.shape[0], -1)  # (bs, 2048)
        x4_max = self.maxpool(x4)
        x4_max = x4_max.view(x4_max.shape[0], -1)  # (bs, 2048)
        x4_avg = self.avgpool(x4)
        x4_avg = x4_avg.view(x4_avg.shape[0], -1)  # (bs, 2048)
        x4_feat = x4_max + x4_avg
        # x4_feat = self.gap(x4)  # (bs, 2048, 1, 1)
        # x4_feat = x4_feat.view(x4_feat.shape[0], -1)  # (bs, 2048)

        if self.training:
            cls_out3 = self.classifier1(x3_feat)
            cls_out4 = self.classifier2(x4_feat)
            return cls_out3, cls_out4, x3_max, x3_avg, x4_max, x4_avg
        else:
            x3_feat = self.classifier1(x3_feat)
            x4_feat = self.classifier2(x4_feat)
            return torch.cat((x3_feat, x4_feat), dim=1)

    def getLoss(self, outputs, labels, **kwargs):
        cls_out3, cls_out4, x3_max, x3_avg, x4_max, x4_avg = outputs

        tri_loss = (TripletLoss(margin=0.3)(x3_max, labels, normalize_feature=False)[0]
                    + TripletLoss(margin=0.3)(x3_avg, labels, normalize_feature=False)[0]
                    + TripletLoss(margin=0.3)(x4_max, labels, normalize_feature=False)[0]
                    + TripletLoss(margin=0.3)(x4_avg, labels, normalize_feature=False)[0]) / 4
        softmax_loss = (CrossEntropyLabelSmooth(self.num_classes)(cls_out3, labels) +
                        CrossEntropyLabelSmooth(self.num_classes)(cls_out4, labels)) / 2
        # softmax_loss = F.cross_entropy(cls_out, labels)

        self.loss = softmax_loss + tri_loss
        # self.loss = softmax_loss
        # return {'Softmax': softmax_loss, 'AM_Softmax': AM_softmax, 'Triplet_loss': tri_loss}
        return {
            'Softmax': softmax_loss,
            'Triplet_loss': tri_loss,
        }

    def load_params_wo_fc(self, state_dict):
        if 'classifier.weight' in state_dict:
            state_dict.pop('classifier.weight')
        if 'amsoftmax.weight' in state_dict:
            state_dict.pop('amsoftmax.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print(f'missing keys {res.missing_keys}')
        print(f'unexpected keys {res.unexpected_keys}')
        # assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    def unfreeze_all_layers(self, ):
        self.train()
        for p in self.parameters():
            p.requires_grad_()

    def unfreeze_specific_layer(self, names):
        if isinstance(names, str):
            names = [names]

        for name, module in self.named_children():
            if name in names:
                module.train()
                for p in module.parameters():
                    p.requires_grad_()
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
