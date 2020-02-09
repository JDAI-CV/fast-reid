# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, num_channels, 1, 1)
        self.register_buffer('pixel_mean', pixel_mean)
        pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, num_channels, 1, 1)
        self.register_buffer('pixel_std', pixel_std)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

        self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)

    def forward(self, inputs, labels=None):
        inputs = self.normalizer(inputs)
        # images = self.preprocess_image(batched_inputs)

        global_feat = self.backbone(inputs)  # (bs, 2048, 16, 8)
        if self.training:
            outputs = self.heads(global_feat, labels)
            return outputs
        else:
            pred_features = self.heads(global_feat)
            return pred_features

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
