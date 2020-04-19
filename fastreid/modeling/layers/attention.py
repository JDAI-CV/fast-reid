# encoding: utf-8
"""
@author:  CASIA IVA
@contact: jliu@nlpr.ia.ac.cn
"""

import torch
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn

__all__ = ['PAM_Module', 'CAM_Module', 'DANetHead',]


class DANetHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_layer: nn.Module,
                 module_class: type,
                 dim_collapsion: int=2):
        super(DANetHead, self).__init__()

        inter_channels = in_channels // dim_collapsion

        self.conv5c = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                3,
                padding=1,
                bias=False
            ),
            norm_layer(inter_channels),
            nn.ReLU()
        )

        self.attention_module = module_class(inter_channels)
        self.conv52 = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                inter_channels,
                3,
                padding=1,
                bias=False
            ),
            norm_layer(inter_channels),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):

        feat2 = self.conv5c(x)
        sc_feat = self.attention_module(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        return sc_output


class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.value_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


# def get_attention_module_instance(
#     name: 'cam | pam | identity',
#     dim: int,
#     *,
#     out_dim=None,
#     use_head: bool=False,
#     dim_collapsion=2  # Used iff `used_head` set to True
# ):
#
#     name = name.lower()
#     assert name in ('cam', 'pam', 'identity')
#
#     module_class = name_module_class_mapping[name]
#
#     if out_dim is None:
#         out_dim = dim
#
#     if use_head:
#         return DANetHead(
#             dim, out_dim,
#             nn.BatchNorm2d,
#             module_class,
#             dim_collapsion=dim_collapsion
#         )
#     else:
#         return module_class(dim)