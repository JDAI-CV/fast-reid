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

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.triplet_loss = TripletLoss(margin)
        
    def forward(self, out, target):
        scores, feats = out
        loss = 0
        if 'softmax' in self.lossType:
            if len(target.size()) == 2:
                loss1, loss2 = self.ce_loss(scores, target[:,0].long()), self.ce_loss(scores, target[:,1].long())
                d = loss1 * target[:,2] + loss2 * (1-target[:,2])
            else:
                d = self.ce_loss(scores, target)
            loss += d.mean()
        if 'triplet' in self.lossType:  
            if len(target.size()) == 2: loss += self.triplet_loss(feats, target[:,0].long())[0]
            else:                       loss += self.triplet_loss(feats, target)[0]

        return loss
