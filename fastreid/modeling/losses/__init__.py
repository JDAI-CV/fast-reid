# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_criterion, LOSS_REGISTRY

from .cross_entroy_loss import CrossEntropyLoss
from .margin_loss import TripletLoss
