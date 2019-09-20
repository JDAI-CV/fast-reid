# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
from torch.utils import model_zoo

from ops import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

model_layers = {
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3]
}

__all__ = ['ResNet', 'Bottleneck']


class IBN(nn.Module):
    """
    IBN with BN:IN = 7:1
    """
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/8)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(torch.cat(split[1:], dim=1).contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, with_ibn=False, gcb=None, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.with_gcb = gcb is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        if with_ibn: self.bn1 = IBN(planes)
        else:        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # GCNet
        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

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

        if self.with_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride, with_ibn, gcb, stage_with_gcb, block, layers):
        scale = 64
        self.inplanes = scale
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0], with_ibn=with_ibn, 
                                       gcb=gcb if stage_with_gcb[0] else None)
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2, with_ibn=with_ibn, 
                                       gcb=gcb if stage_with_gcb[1] else None)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2, with_ibn=with_ibn, 
                                       gcb=gcb if stage_with_gcb[2] else None)
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=last_stride, 
                                       gcb=gcb if stage_with_gcb[3] else None)

    def _make_layer(self, block, planes, blocks, stride=1, with_ibn=False, gcb=None):
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
        layers.append(block(self.inplanes, planes, with_ibn, gcb, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_ibn, gcb))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_pretrain(self, model_path=''):
        with_model_path = (model_path is not '')
        if not with_model_path:  # resnet pretrain
            state_dict = model_zoo.load_url(model_urls[self._model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.load_state_dict(state_dict)
        else:
            # ibn pretrain
            state_dict = torch.load(model_path)['state_dict']
            state_dict.pop('module.fc.weight')
            state_dict.pop('module.fc.bias')
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])  # remove module in name
                if self.state_dict()[new_k].shape == state_dict[k].shape:
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            self.load_state_dict(state_dict, strict=False)

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @classmethod
    def from_name(cls, model_name, last_stride, with_ibn, gcb, stage_with_gcb):
        cls._model_name = model_name
        return ResNet(last_stride, with_ibn, gcb, stage_with_gcb, block=Bottleneck, layers=model_layers[model_name])