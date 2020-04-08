# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.modeling.backbones import *
from fastreid.modeling.model_utils import *
from fastreid.modeling.heads import *
from fastreid.layers import NoBiasBatchNorm1d, GeM


class MaskUnit(nn.Module):
    def __init__(self, in_planes=2048):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.mask = nn.Linear(in_planes, 1, bias=None)

    def forward(self, x):
        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x)
        xx = x.view(x.size(0), x.size(1), -1)  # (bs, 2048, 192)
        x1 = x1.view(x1.size(0), x1.size(1), -1)  # (bs, 2048, 48)
        x2 = x2.view(x2.size(0), x2.size(1), -1)  # (bs, 2048, 33)
        feat = torch.cat((xx, x1, x2), dim=2)  # (bs, 2048, 273)
        feat = feat.transpose(1, 2)  # (bs, 274, 2048)
        mask_scores = self.mask(feat)  # (bs, 274, 1)
        scores = F.normalize(mask_scores[:, :192], p=1, dim=1)  # (bs, 192, 1)
        mask_feat = torch.bmm(xx, scores)  # (bs, 2048, 1)
        return mask_feat.squeeze(2), mask_scores.squeeze(2)


class Maskmodel(nn.Module):
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
        self.num_classes = num_classes
        # self.gap = GeM()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.res_part = Bottleneck(2048, 512)

        self.global_reduction = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )
        self.global_bnneck = NoBiasBatchNorm1d(1024)
        self.global_bnneck.apply(weights_init_kaiming)
        self.global_fc = nn.Linear(1024, self.num_classes, bias=False)
        self.global_fc.apply(weights_init_classifier)

        self.mask_layer = MaskUnit(self.in_planes)
        self.mask_reduction = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )
        self.mask_bnneck = NoBiasBatchNorm1d(1024)
        self.mask_bnneck.apply(weights_init_kaiming)

        self.mask_fc = nn.Linear(1024, self.num_classes, bias=False)
        self.mask_fc.apply(weights_init_classifier)

    def forward(self, x, label=None, pose=None):
        global_feat = self.base(x)  # (bs, 2048, 24, 8)
        pool_feat = self.gap(global_feat)  # (bs, 2048, 1, 1)
        pool_feat = pool_feat.view(-1, 2048)  # (bs, 2048)
        re_feat = self.global_reduction(pool_feat)  # (bs, 1024)
        bn_re_feat = self.global_bnneck(re_feat)  # normalize for angular softmax

        # global_feat = global_feat.view(global_feat.size(0), global_feat.size(1), -1)
        # pose = pose.unsqueeze(2)
        # pose_feat = torch.bmm(global_feat, pose).squeeze(2)  # (bs, 2048)
        # fused_feat = pool_feat + pose_feat
        # bn_feat = self.bottleneck(fused_feat)
        # mask_feat = self.res_part(global_feat)
        mask_feat, mask_scores = self.mask_layer(global_feat)
        mask_re_feat = self.mask_reduction(mask_feat)
        bn_mask_feat = self.mask_bnneck(mask_re_feat)
        if self.training:
            cls_out = self.global_fc(bn_re_feat)
            mask_cls_out = self.mask_fc(bn_mask_feat)
            # am_out = self.amsoftmax(feat, label)
            return cls_out, mask_cls_out, pool_feat, mask_feat, mask_scores
        else:
            return torch.cat((bn_re_feat, bn_mask_feat), dim=1), bn_mask_feat

    def getLoss(self, outputs, labels, mask_labels, **kwargs):
        cls_out, mask_cls_out, feat, mask_feat, mask_scores = outputs
        # cls_out, feat = outputs

        tri_loss = (TripletLoss(margin=-1)(feat, labels, normalize_feature=False)[0] +
                    TripletLoss(margin=-1)(mask_feat, labels, normalize_feature=False)[0]) / 2
        # mask_feat_tri_loss = TripletLoss(margin=-1)(mask_feat, labels, normalize_feature=False)[0]
        softmax_loss = (F.cross_entropy(cls_out, labels) + F.cross_entropy(mask_cls_out, labels)) / 2
        mask_loss = nn.functional.mse_loss(mask_scores, mask_labels) * 0.16

        self.loss = softmax_loss + tri_loss + mask_loss
        # self.loss = softmax_loss + tri_loss + mask_loss
        return {
            'softmax': softmax_loss,
            'tri': tri_loss,
            'mask': mask_loss,
        }

    def load_params_wo_fc(self, state_dict):
        state_dict.pop('global_fc.weight')
        state_dict.pop('mask_fc.weight')
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
