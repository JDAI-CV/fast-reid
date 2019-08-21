# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .triplet_loss import TripletLoss


__all__ = ['reidLoss']


class reidLoss(nn.Module):
    def __init__(self, lossType:list, margin:float):
        super().__init__()
        self.lossType = lossType

        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin)
        
    def forward(self, out, target):
        scores, feats = out
        loss = 0
        if 'softmax' in self.lossType:       loss += self.ce_loss(scores, target)
        if 'triplet' in self.lossType:       loss += self.triplet_loss(feats, target)[0]

        return loss
