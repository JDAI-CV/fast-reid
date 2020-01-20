# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# Ref:
# @author: wujiyang
# @contact: wujiyang@hust.edu.cn
# @file: attention.py
# @time: 2019/2/14 14:12
# @desc: Residual Attention Network for Image Classification, CVPR 2017.
#        Attention 56 and Attention 92.


import torch
import torch.nn as nn
import numpy as np
import sys


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.res_bottleneck = nn.Sequential(nn.BatchNorm2d(in_channel),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channel, out_channel//4, 1, 1, bias=False),
                                            nn.BatchNorm2d(out_channel//4),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channel//4, out_channel//4, 3, stride, padding=1, bias=False),
                                            nn.BatchNorm2d(out_channel//4),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channel//4, out_channel, 1, 1, bias=False))
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1, stride, bias=False)

    def forward(self, x):
        res = x
        out = self.res_bottleneck(x)
        if self.in_channel != self.out_channel or self.stride != 1:
            res = self.shortcut(x)

        out += res
        return out


class AttentionModule_stage1(nn.Module):

    # input size is 56*56
    def __init__(self, in_channel, out_channel, size1=(128, 64), size2=(64, 32), size3=(32, 16)):
        super(AttentionModule_stage1, self).__init__()
        self.share_residual_block = ResidualBlock(in_channel, out_channel)
        self.trunk_branches = nn.Sequential(ResidualBlock(in_channel, out_channel),
                                            ResidualBlock(in_channel, out_channel))

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block1 = ResidualBlock(in_channel, out_channel)
        self.skip_connect1 = ResidualBlock(in_channel, out_channel)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block2 = ResidualBlock(in_channel, out_channel)
        self.skip_connect2 = ResidualBlock(in_channel, out_channel)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block3 = nn.Sequential(ResidualBlock(in_channel, out_channel),
                                         ResidualBlock(in_channel, out_channel))

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.mask_block4 = ResidualBlock(in_channel, out_channel)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.mask_block5 = ResidualBlock(in_channel, out_channel)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.mask_block6 = nn.Sequential(nn.BatchNorm2d(out_channel),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channel, out_channel, 1, 1, bias=False),
                                         nn.BatchNorm2d(out_channel),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channel, out_channel, 1, 1, bias=False),
                                         nn.Sigmoid())

        self.last_block = ResidualBlock(in_channel, out_channel)

    def forward(self, x):
        x = self.share_residual_block(x)
        out_trunk = self.trunk_branches(x)

        out_pool1 = self.mpool1(x)
        out_block1 = self.mask_block1(out_pool1)
        out_skip_connect1 = self.skip_connect1(out_block1)

        out_pool2 = self.mpool2(out_block1)
        out_block2 = self.mask_block2(out_pool2)
        out_skip_connect2 = self.skip_connect2(out_block2)

        out_pool3 = self.mpool3(out_block2)
        out_block3 = self.mask_block3(out_pool3)
        #
        out_inter3 = self.interpolation3(out_block3) + out_block2
        out = out_inter3 + out_skip_connect2
        out_block4 = self.mask_block4(out)

        out_inter2 = self.interpolation2(out_block4) + out_block1
        out = out_inter2 + out_skip_connect1
        out_block5 = self.mask_block5(out)

        out_inter1 = self.interpolation1(out_block5) + out_trunk
        out_block6 = self.mask_block6(out_inter1)

        out = (1 + out_block6) + out_trunk
        out_last = self.last_block(out)

        return out_last


class AttentionModule_stage2(nn.Module):

    # input image size is 28*28
    def __init__(self, in_channels, out_channels, size1=(64, 32), size2=(32, 16)):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax4_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        out = out_interp2 + out_skip1_connection

        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage3(nn.Module):

    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size1=(32, 16)):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class ResidualAttentionNet_56(nn.Module):

    # for input size 112
    def __init__(self, feature_dim=512, drop_ratio=0.4):
        super(ResidualAttentionNet_56, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 512, 2)
        self.attention_module3 = AttentionModule_stage3(512, 512)
        self.residual_block4 = ResidualBlock(512, 512, 2)
        self.residual_block5 = ResidualBlock(512, 512)
        self.residual_block6 = ResidualBlock(512, 512)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 16 * 8, feature_dim),)
                                          # nn.BatchNorm1d(feature_dim))

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.output_layer(out)

        return out


class ResidualAttentionNet_92(nn.Module):

    # for input size 112
    def __init__(self, feature_dim=512, drop_ratio=0.4):
        super(ResidualAttentionNet_92, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(2048),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(2048 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.output_layer(out)

        return out

