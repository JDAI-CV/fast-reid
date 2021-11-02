# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 23:06:51
# @Author  : zuchen.wang@vipshop.com
# @File    : senet.py
import logging
import math
from typing import Tuple

import pretrainedmodels
import torch
from torch import nn

from fastreid.config.config import CfgNode
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


class SePcbNet(nn.Module):
	def __init__(self,
	             part_num: int,
	             embedding_dim: int,
	             part_dim: int,
	             last_stride: int,
	             ):
		super(SePcbNet, self).__init__()
		self.part_num = part_num
		self.embedding_dim = embedding_dim
		self.part_dim = part_dim
		self.last_stride = (last_stride, last_stride)

		self.cnn = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained='imagenet')
		self.cnn.layer4[0].downsample[0].stride = self.last_stride
		self.cnn.layer4[0].conv2.stride = self.last_stride
		self.cnn.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.avg_pool_part6 = nn.AdaptiveAvgPool2d((self.part_num, 1))

		for i in range(self.part_num):
			setattr(self, 'reduction_' + str(i),
			        nn.Sequential(
				        nn.Conv2d(self.embedding_dim, self.part_dim, (1, 1), bias=False),
				        nn.BatchNorm2d(self.part_dim),
				        nn.ReLU()
			        ))

		self.random_init()

	def forward(self, x):
		x = self.cnn.layer0(x)
		x = self.cnn.layer1(x)
		x = self.cnn.layer2(x)
		x = self.cnn.layer3(x)
		x = self.cnn.layer4(x)

		x_full = self.cnn.avg_pool(x)
		x_full = x_full.reshape(x_full.shape[0], -1)

		x_part = self.avg_pool_part6(x)
		parts = []

		for i in range(self.part_num):
			reduction_i = getattr(self, 'reduction_' + str(i))
			part_i = x_part[:, :, i: i + 1, :]
			parts.append(reduction_i(part_i).squeeze())

		return {
			'full': x_full,
			'parts': parts,
		}

	def random_init(self):
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

		self.cnn.layer0.conv1.weight.data.normal_(0, math.sqrt(2. / (7 * 7 * 64)))


@BACKBONE_REGISTRY.register()
def build_senet_pcb_backbone(cfg: CfgNode):
	# fmt: off
	pretrain = cfg.MODEL.BACKBONE.PRETRAIN
	pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
	last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
	part_num = cfg.MODEL.PCB.PART_NUM
	part_dim = cfg.MODEL.PCB.PART_DIM
	embedding_dim = cfg.MODEL.PCB.EMBEDDING_DIM
	# fmt: on

	model = SePcbNet(part_num=part_num, embedding_dim=embedding_dim, part_dim=part_dim, last_stride=last_stride)

	if pretrain:
		if pretrain_path:
			try:
				state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
				new_state_dict = {}
				for k in state_dict:
					new_k = 'cnn.' + k
					if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
						new_state_dict[new_k] = state_dict[k]
				state_dict = new_state_dict
				logger.info(f"Loading pretrained model from {pretrain_path}")
			except FileNotFoundError as e:
				logger.error(f'{pretrain_path} is not found! Please check this path.')
				raise e
			except KeyError as e:
				logger.error("State dict keys error! Please check the state dict.")
				raise e
		else:
			logger.info('Not config pretrained mode with SePcbNet, the weights will be random init')

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
