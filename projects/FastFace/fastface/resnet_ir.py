# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from collections import namedtuple

from torch import nn

from fastreid.layers import get_norm, SELayer
from fastreid.modeling.backbones import BACKBONE_REGISTRY


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, bn_norm, stride, with_se=False):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                get_norm(bn_norm, depth))
        self.res_layer = nn.Sequential(
            get_norm(bn_norm, in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            get_norm(bn_norm, depth),
            SELayer(depth, 16) if with_se else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "bn_norm", "stride", "with_se"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, bn_norm, num_units, with_se, stride=2):
    return [Bottleneck(in_channel, depth, bn_norm, stride, with_se)] + \
           [Bottleneck(depth, depth, bn_norm, 1, with_se) for _ in range(num_units - 1)]


def get_blocks(bn_norm, with_se, num_layers):
    if num_layers == "50x":
        blocks = [
            get_block(in_channel=64, depth=64, bn_norm=bn_norm, num_units=3, with_se=with_se),
            get_block(in_channel=64, depth=128, bn_norm=bn_norm, num_units=4, with_se=with_se),
            get_block(in_channel=128, depth=256, bn_norm=bn_norm, num_units=14, with_se=with_se),
            get_block(in_channel=256, depth=512, bn_norm=bn_norm, num_units=3, with_se=with_se)
        ]
    elif num_layers == "100x":
        blocks = [
            get_block(in_channel=64, depth=64, bn_norm=bn_norm, num_units=3, with_se=with_se),
            get_block(in_channel=64, depth=128, bn_norm=bn_norm, num_units=13, with_se=with_se),
            get_block(in_channel=128, depth=256, bn_norm=bn_norm, num_units=30, with_se=with_se),
            get_block(in_channel=256, depth=512, bn_norm=bn_norm, num_units=3, with_se=with_se)
        ]
    elif num_layers == "152x":
        blocks = [
            get_block(in_channel=64, depth=64, bn_norm=bn_norm, num_units=3, with_se=with_se),
            get_block(in_channel=64, depth=128, bn_norm=bn_norm, num_units=8, with_se=with_se),
            get_block(in_channel=128, depth=256, bn_norm=bn_norm, num_units=36, with_se=with_se),
            get_block(in_channel=256, depth=512, bn_norm=bn_norm, num_units=3, with_se=with_se)
        ]
    return blocks


class ResNetIR(nn.Module):
    def __init__(self, num_layers, bn_norm, drop_ratio, with_se):
        super(ResNetIR, self).__init__()
        assert num_layers in ["50x", "100x", "152x"], "num_layers should be 50,100, or 152"
        blocks = get_blocks(bn_norm, with_se, num_layers)
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         get_norm(bn_norm, 64),
                                         nn.PReLU(64))
        self.output_layer = nn.Sequential(get_norm(bn_norm, 512),
                                          nn.Dropout(drop_ratio))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR(bottleneck.in_channel,
                                  bottleneck.depth,
                                  bottleneck.bn_norm,
                                  bottleneck.stride,
                                  bottleneck.with_se))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


@BACKBONE_REGISTRY.register()
def build_resnetIR_backbone(cfg):
    """
    Create a ResNetIR instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    bn_norm = cfg.MODEL.BACKBONE.NORM
    with_se = cfg.MODEL.BACKBONE.WITH_SE
    depth   = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    model = ResNetIR(depth, bn_norm, 0.5, with_se)
    return model
