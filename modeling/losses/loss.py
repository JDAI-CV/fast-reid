# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .center_loss import CenterLoss
from .cosface import AddMarginProduct
from .label_smooth import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss

__all__ = ['reidLoss']


class reidLoss(nn.Module):
    def __init__(self, lossType: list, margin: float, num_classes: float):
        super().__init__()
        self.lossType = lossType

        if 'softmax' in self.lossType:          self.ce_loss = nn.CrossEntropyLoss()
        if 'softmax_smooth' in self.lossType:   self.ce_loss = CrossEntropyLabelSmooth(num_classes)
        if 'triplet' in self.lossType:          self.triplet_loss = TripletLoss(margin)
        # if 'center' in self.lossType:           self.center_loss = CenterLoss(num_classes, feat_dim)
        
    def forward(self, out, labels):
        cls_scores, feats = out
        loss = 0
        if 'softmax' or 'softmax_smooth' in self.lossType:  loss += self.ce_loss(cls_scores, labels)
        if 'triplet' in self.lossType:                      loss += self.triplet_loss(feats, labels)[0]
        # if 'center' in self.lossType:                       loss += 0.0005 * self.center_loss(feats, labels)
        return loss
