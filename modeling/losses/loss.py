# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .triplet_loss import TripletLoss
from .label_smooth import CrossEntropyLabelSmooth

__all__ = ['reidLoss']


class reidLoss(nn.Module):
    def __init__(self, lossType:list, margin:float, num_classes:float):
        super().__init__()
        self.lossType = lossType

        if 'softmax' in self.lossType:        self.ce_loss = nn.CrossEntropyLoss()
        if 'softmax_smooth' in self.lossType: self.ce_loss = CrossEntropyLabelSmooth(num_classes)
        if 'triplet' in self.lossType:        self.triplet_loss = TripletLoss(margin)
        
    def forward(self, out, target):
        scores, feats = out
        loss = 0
        if 'softmax' or 'softmax_smooth' in self.lossType: loss += self.ce_loss(scores, target)
        if 'triplet' in self.lossType:                     loss += self.triplet_loss(feats, target)[0]
        return loss
