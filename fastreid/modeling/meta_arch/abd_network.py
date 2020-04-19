# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads
from ..model_utils import weights_init_kaiming
from fastreid.modeling.layers import CAM_Module, PAM_Module, DANetHead, Flatten


@META_ARCH_REGISTRY.register()
class ABD_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        backbone = build_backbone(cfg)
        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.shallow_cam = CAM_Module(256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3
        )

        # global branch
        self.global_res4 = copy.deepcopy(backbone.layer4)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            # reduce
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.global_branch.apply(weights_init_kaiming)

        self.global_head = build_reid_heads(cfg, 1024, nn.Identity())

        # attention branch
        self.att_res4 = copy.deepcopy(backbone.layer4)
        # reduce
        self.att_reduce = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.att_reduce.apply(weights_init_kaiming)

        self.abd_branch = ABDBranch(1024)
        self.abd_branch.apply(weights_init_kaiming)

        self.att_head = build_reid_heads(cfg, 1024, nn.Identity())

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]

        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs["camid"]

        feat = self.backbone1(images)
        feat = self.shallow_cam(feat)
        feat = self.backbone2(feat)

        # global branch
        global_feat = self.global_res4(feat)
        global_feat = self.global_branch(global_feat)
        global_logits, global_feat = self.global_head(global_feat, targets)

        # attention branch
        att_feat = self.att_res4(feat)
        att_feat = self.att_reduce(att_feat)
        att_feat = self.abd_branch(att_feat)
        att_logits, att_feat = self.att_bnneck(att_feat, targets)

        return global_logits, global_feat, att_logits, att_feat, targets

    def losses(self, outputs):
        loss_dict = {}
        loss_dict.update(self.global_head.losses(self._cfg, outputs[0], outputs[1], outputs[-1], 'global_'))
        loss_dict.update(self.att_head.losses(self._cfg, outputs[2], outputs[3], outputs[-1], 'att_'))
        return loss_dict

    def inference(self, images):
        assert not self.training
        feat = self.backbone1(images)
        feat = self.shallow_cam(feat)
        feat = self.backbone2(feat)

        # global branch
        global_feat = self.global_res4(feat)
        global_feat = self.global_branch(global_feat)
        global_pred_feat = self.global_head(global_feat)

        # attention branch
        att_feat = self.att_res4(feat)
        att_feat = self.att_reduce(att_feat)
        att_feat = self.abd_branch(att_feat)
        att_pred_feat = self.att_head(att_feat)

        pred_feat = torch.cat([global_pred_feat, att_pred_feat], dim=1)
        return F.normalize(pred_feat)


class ABDBranch(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1024
        self.part_num = 2
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten())

        self._init_attention_modules()

    def _init_attention_modules(self):
        self.before_module = DANetHead(self.output_dim, self.output_dim, nn.BatchNorm2d, nn.Identity)

        self.cam_module = DANetHead(self.output_dim, self.output_dim, nn.BatchNorm2d, CAM_Module)

        self.pam_module = DANetHead(self.output_dim, self.output_dim, nn.BatchNorm2d, PAM_Module)

        self.sum_conv = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1)
        )

    def forward(self, x):
        assert x.size(2) % self.part_num == 0, \
            "Height {} is not a multiplication of {}. Aborted.".format(x.size(2), self.part_num)

        before_x = self.before_module(x)
        cam_x = self.cam_module(x)
        pam_x = self.pam_module(x)
        sum_x = before_x + cam_x + pam_x
        att_feat = self.sum_conv(sum_x)
        avg_feat = self.avg_pool(att_feat)
        return avg_feat
        # margin = x.size(2) // self.part_num
        # for p in range(self.part_num):
        #     x_sliced = x[:, :, margin * p:margin * (p + 1), :]
        #
        #     to_sum = []
        #     # module_name: str
        #     for module_name in self.dan_module_names:
        #         x_out = getattr(self, module_name)(x_sliced)
        #         to_sum.append(x_out)
        #         fmap[module_name.partition('_')[0]].append(x_out)
        #
        #     fmap_after = self.sum_conv(sum(to_sum))
        #     fmap['after'].append(fmap_after)
        #
        #     v = self.avgpool(fmap_after)
        #     v = v.view(v.size(0), -1)
        #     triplet.append(v)
        #     predict.append(v)
        #     v = self.classifiers[p](v)
        #     xent.append(v)
        #
        # return predict, xent, triplet, fmap
