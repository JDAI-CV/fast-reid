# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

# based on:
# https://github.com/XingangPan/IBN-Net/blob/master/models/imagenet/resnext_ibn_a.py

import math
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from ...layers import IBN
from .build import BACKBONE_REGISTRY


class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, with_ibn, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        if with_ibn:
            self.bn1 = IBN(D * C)
        else:
            self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, last_stride, with_ibn, block, layers, baseWidth=4, cardinality=32):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_ibn=with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, with_ibn=with_ibn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, with_ibn=with_ibn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, with_ibn=with_ibn)

        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, with_ibn=False):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, with_ibn, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_ibn, self.baseWidth, self.cardinality, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def random_init(self):
        self.conv1.weight.data.normal_(0, math.sqrt(2. / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


@BACKBONE_REGISTRY.register()
def build_resnext_backbone(cfg):
    """
    Create a ResNeXt instance from config.
    Returns:
        ResNeXt: a :class:`ResNeXt` instance.
    """

    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
    with_ibn = cfg.MODEL.BACKBONE.WITH_IBN
    with_se = cfg.MODEL.BACKBONE.WITH_SE
    with_nl = cfg.MODEL.BACKBONE.WITH_NL
    depth = cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {50: [0, 2, 3, 0], 101: [0, 2, 3, 0]}[depth]
    model = ResNeXt(last_stride, with_ibn, Bottleneck, num_blocks_per_stage)
    if pretrain:
        # if not with_ibn:
        # original resnet
        # state_dict = model_zoo.load_url(model_urls[depth])
        # else:
        # ibn resnet
        state_dict = torch.load(pretrain_path)['state_dict']
        # remove module in name
        new_state_dict = {}
        for k in state_dict:
            new_k = '.'.join(k.split('.')[1:])
            if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                new_state_dict[new_k] = state_dict[k]
        state_dict = new_state_dict
        res = model.load_state_dict(state_dict, strict=False)
        logger = logging.getLogger(__name__)
        logger.info('missing keys is {}'.format(res.missing_keys))
        logger.info('unexpected keys is {}'.format(res.unexpected_keys))
    return model
