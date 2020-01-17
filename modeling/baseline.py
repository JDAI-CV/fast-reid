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
from layers import bn_no_bias, GeM, ClassBlock


class Baseline(nn.Module):
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
            # self.base = InsResNet50()
            # state_dict = torch.load(model_path)['model']
            # res = self.base.load_state_dict(state_dict, strict=False)
            # print(res.missing_keys)
            # print(res.unexpected_keys)
            self.in_planes = 2048
        elif 'osnet' in backbone:
            if with_ibn:
                self.base = osnet_ibn_x1_0(pretrained=pretrain)
            else:
                self.base = osnet_x1_0(pretrained=pretrain)
            self.in_planes = 512
        elif 'attention' in backbone:
            self.base = ResidualAttentionNet_56(feature_dim=512)
        else:
            print(f'not support {backbone} backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = GeM()
        self.num_classes = num_classes

        self.bottleneck = bn_no_bias(2048)
        self.bottleneck.apply(weights_init_kaiming)

        # self.classifier = ClassBlock(self.in_planes, self.num_classes)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        # self.classifier = CircleLoss(self.in_planes, self.num_classes, s=256.0, m=0.25)

    def forward(self, x, label=None, **kwargs):
        global_feat = self.base(x)  # (bs, 2048, 16, 8)
        global_feat = self.gap(global_feat)  # (bs, 2048, 1, 1)
        global_feat = global_feat.view(-1, 2048)
        bn_feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_out = self.classifier(bn_feat)
            # cls_out = self.classifier(pool_feat, label)
            # return cls_out, mask_out, pool_feat, mask_feat, mask_score
            return cls_out, global_feat
        else:
            return bn_feat

    def getLoss(self, outputs, labels, **kwargs):
        cls_out, feat = outputs

        tri_loss = TripletLoss(margin=-1)(feat, labels, normalize_feature=False)[0]
        # tri_loss = 0.4*RankedLoss(1.3, 2.0, 1.0)(feat, labels, normalize_features=True)
        # softmax_loss = CrossEntropyLabelSmooth(self.num_classes)(cls_out, labels)
        softmax_loss = F.cross_entropy(cls_out, labels)

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
