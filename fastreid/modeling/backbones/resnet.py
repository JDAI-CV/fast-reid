# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn

from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm


logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion, num_splits)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64, num_splits)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, num_splits, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, num_splits, with_se=with_se)

        self.random_init()

        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", num_splits=1, with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion, num_splits),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, num_splits):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm, num_splits) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm, num_splits) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm, num_splits) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm, num_splits) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = cfg.MODEL.BACKBONE.NORM
    num_splits = cfg.MODEL.BACKBONE.NORM_SPLIT
    with_ibn = cfg.MODEL.BACKBONE.WITH_IBN
    with_se = cfg.MODEL.BACKBONE.WITH_SE
    with_nl = cfg.MODEL.BACKBONE.WITH_NL
    depth = cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = ResNet(last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model
