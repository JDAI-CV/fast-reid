# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
from torch import nn

from .backbones import ResNet, Bottleneck
from .utils import *


class MGN(nn.Module):
    in_planes = 2048
    feats = 256

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
            base_module = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            base_module.load_pretrain(model_path)

        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            base_module.conv1,
            base_module.bn1,
            base_module.relu,
            base_module.maxpool,
            base_module.layer1,
            base_module.layer2,
            base_module.layer3[0]
        )
        
        res_conv4 = nn.Sequential(*base_module.layer3[1:])
        
        res_g_conv5 = base_module.layer4
        
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False),
                                                           nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512)
        )
        res_p_conv5.load_state_dict(base_module.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool_zp2 = nn.MaxPool2d((12, 9))
        self.maxpool_zp3 = nn.MaxPool2d((8, 9))

        self.reduction = nn.Conv2d(2048, self.feats, 1, bias=False)
        self.bn_neck = BN_no_bias(self.feats)
        # self.bn_neck_2048_0 = BN_no_bias(self.feats)
        # self.bn_neck_2048_1 = BN_no_bias(self.feats)
        # self.bn_neck_2048_2 = BN_no_bias(self.feats)
        # self.bn_neck_256_1_0 = BN_no_bias(self.feats)
        # self.bn_neck_256_1_1 = BN_no_bias(self.feats)
        # self.bn_neck_256_2_0 = BN_no_bias(self.feats)
        # self.bn_neck_256_2_1 = BN_no_bias(self.feats)
        # self.bn_neck_256_2_2 = BN_no_bias(self.feats)

        self.fc_id_2048_0 = nn.Linear(self.feats, self.num_classes, bias=False)
        self.fc_id_2048_1 = nn.Linear(self.feats, self.num_classes, bias=False)
        self.fc_id_2048_2 = nn.Linear(self.feats, self.num_classes, bias=False)

        self.fc_id_256_1_0 = nn.Linear(self.feats, self.num_classes, bias=False)
        self.fc_id_256_1_1 = nn.Linear(self.feats, self.num_classes, bias=False)
        self.fc_id_256_2_0 = nn.Linear(self.feats, self.num_classes, bias=False)
        self.fc_id_256_2_1 = nn.Linear(self.feats, self.num_classes, bias=False)
        self.fc_id_256_2_2 = nn.Linear(self.feats, self.num_classes, bias=False)

        self.fc_id_2048_0.apply(weights_init_classifier)
        self.fc_id_2048_1.apply(weights_init_classifier)
        self.fc_id_2048_2.apply(weights_init_classifier)
        self.fc_id_256_1_0.apply(weights_init_classifier)
        self.fc_id_256_1_1.apply(weights_init_classifier)
        self.fc_id_256_2_0.apply(weights_init_classifier)
        self.fc_id_256_2_1.apply(weights_init_classifier)
        self.fc_id_256_2_2.apply(weights_init_classifier)

    def forward(self, x, label=None):
        global_feat = self.backbone(x)

        p1 = self.p1(global_feat)  # (bs, 2048, 18, 9)
        p2 = self.p2(global_feat)  # (bs, 2048, 18, 9)
        p3 = self.p3(global_feat)  # (bs, 2048, 18, 9)

        zg_p1 = self.avgpool(p1)  # (bs, 2048, 1, 1)
        zg_p2 = self.avgpool(p2)  # (bs, 2048, 1, 1)
        zg_p3 = self.avgpool(p3)  # (bs, 2048, 1, 1)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        g_p1 = zg_p1.squeeze(3).squeeze(2)  # (bs, 2048)
        fg_p1 = self.reduction(zg_p1).squeeze(3).squeeze(2)
        bn_fg_p1 = self.bn_neck(fg_p1)
        g_p2 = zg_p2.squeeze(3).squeeze(2)
        fg_p2 = self.reduction(zg_p2).squeeze(3).squeeze(2)  # (bs, 256)
        bn_fg_p2 = self.bn_neck(fg_p2)
        g_p3 = zg_p3.squeeze(3).squeeze(2)
        fg_p3 = self.reduction(zg_p3).squeeze(3).squeeze(2)
        bn_fg_p3 = self.bn_neck(fg_p3)

        f0_p2 = self.bn_neck(self.reduction(z0_p2).squeeze(3).squeeze(2))
        f1_p2 = self.bn_neck(self.reduction(z1_p2).squeeze(3).squeeze(2))
        f0_p3 = self.bn_neck(self.reduction(z0_p3).squeeze(3).squeeze(2))
        f1_p3 = self.bn_neck(self.reduction(z1_p3).squeeze(3).squeeze(2))
        f2_p3 = self.bn_neck(self.reduction(z2_p3).squeeze(3).squeeze(2))

        if self.training:
            l_p1 = self.fc_id_2048_0(bn_fg_p1)
            l_p2 = self.fc_id_2048_1(bn_fg_p2)
            l_p3 = self.fc_id_2048_2(bn_fg_p3)

            l0_p2 = self.fc_id_256_1_0(f0_p2)
            l1_p2 = self.fc_id_256_1_1(f1_p2)
            l0_p3 = self.fc_id_256_2_0(f0_p3)
            l1_p3 = self.fc_id_256_2_1(f1_p3)
            l2_p3 = self.fc_id_256_2_2(f2_p3)
            return g_p1, g_p2, g_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
            # return g_p2, l_p2, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        else:
            return torch.cat([bn_fg_p1, bn_fg_p2, bn_fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

    def load_params_wo_fc(self, state_dict):
        # state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'
