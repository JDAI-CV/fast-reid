# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .triplet_loss import TripletLoss, RankedLoss
from .label_smooth import CrossEntropyLabelSmooth
from .arcface import ArcCos
from .cosface import AM_softmax
from .circle_loss import CircleLoss