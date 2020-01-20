# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from torch import nn

from .build import META_ARCH_REGISTRY
from ..backbones import *
from ..heads import build_reid_heads


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.MODEL.BACKBONE
        self.last_stride = cfg.MODEL.LAST_STRIDE
        self.with_ibn = cfg.MODEL.WITH_IBN
        self.with_se = cfg.MODEL.WITH_SE
        self.pretrain = cfg.MODEL.PRETRAIN
        self.pretrain_path = cfg.MODEL.PRETRAIN_PATH

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(1, num_channels, 1, 1)
        pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(1, num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        if 'resnet' in self.backbone:
            self.backbone = ResNet.from_name(self.backbone, self.pretrain, self.last_stride, self.with_ibn,
                                             self.with_se, pretrain_path=self.pretrain_path)
            self.in_planes = 2048
        elif 'osnet' in self.backbone:
            if self.with_ibn:
                self.backbone = osnet_ibn_x1_0(pretrained=self.pretrain)
            else:
                self.backbone = osnet_x1_0(pretrained=self.pretrain)
            self.in_planes = 512
        elif 'attention' in self.backbone:
            self.backbone = ResidualAttentionNet_56(feature_dim=512)
        else:
            print(f'not support {self.backbone} backbone')

        # self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        global_feat = self.backbone(images)  # (bs, 2048, 16, 8)
        if self.training:
            labels = torch.stack([torch.tensor(x["targets"]).long().to(self.device) for x in batched_inputs])
            losses = self.heads(global_feat, labels)
            return losses
        else:
            pred_features = self.heads(global_feat)
        return {
            'pred_features': pred_features
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        images = [x["images"] for x in batched_inputs]
        w = images[0].size[0]
        h = images[0].size[1]
        tensor = torch.zeros((len(images), 3, h, w), dtype=torch.uint8)
        for i, image in enumerate(images):
            image = np.asarray(image, dtype=np.uint8)
            numpy_array = np.rollaxis(image, 2)
            tensor[i] += torch.from_numpy(numpy_array)

        tensor = tensor.to(dtype=torch.float32, device=self.device, non_blocking=True)
        tensor = self.normalizer(tensor)
        return tensor

    def load_params_wo_fc(self, state_dict):
        if 'classifier.weight' in state_dict:
            state_dict.pop('classifier.weight')
        if 'amsoftmax.weight' in state_dict:
            state_dict.pop('amsoftmax.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print(f'missing keys {res.missing_keys}')
        print(f'unexpected keys {res.unexpected_keys}')
        # assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    # def unfreeze_all_layers(self, ):
    #     self.train()
    #     for p in self.parameters():
    #         p.requires_grad_()
    #
    # def unfreeze_specific_layer(self, names):
    #     if isinstance(names, str):
    #         names = [names]
    #
    #     for name, module in self.named_children():
    #         if name in names:
    #             module.train()
    #             for p in module.parameters():
    #                 p.requires_grad_()
    #         else:
    #             module.eval()
    #             for p in module.parameters():
    #                 p.requires_grad_(False)
