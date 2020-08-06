import torch
import torch.nn as nn
from collections import OrderedDict

from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from .network import ShuffleNetV2


__all__ = ['build_shufflenetv2_backbone']


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg):

    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    model_size = cfg.MODEL.BACKBONE.MODEL_SIZE

    return ShuffleNetV2Backbone(model_size=model_size, pretrained=pretrain, pretrain_path=pretrain_path)


class ShuffleNetV2Backbone(nn.Module):

    def __init__(self, model_size, pretrained=False, pretrain_path=''):
        super(ShuffleNetV2Backbone, self).__init__()

        model = ShuffleNetV2(model_size=model_size)
        if pretrained:
            new_state_dict = OrderedDict()
            state_dict = torch.load(pretrain_path)['state_dict']
            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    k = k[7:]
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)

        self.backbone = nn.Sequential(
            model.first_conv, model.maxpool, model.features, model.conv_last)

    def forward(self, x):
        return self.backbone(x)


