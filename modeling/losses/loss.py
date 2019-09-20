# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .label_smooth import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss

__all__ = ['reidLoss']


class reidLoss(object):
    def __init__(self, lossType: list, margin: float, num_classes: float):
        super().__init__()
        self.lossType = lossType

        if 'softmax' in self.lossType:          self.ce_loss = nn.CrossEntropyLoss()
        if 'softmax_smooth' in self.lossType:   self.ce_loss = CrossEntropyLabelSmooth(num_classes)
        if 'triplet' in self.lossType:          self.triplet_loss = TripletLoss(margin)
        # if 'center' in self.lossType:           self.center_loss = CenterLoss(num_classes, feat_dim)
        
    def __call__(self, outputs, labels):
        # cls_scores, feats = outputs
        loss = {}
        if 'softmax' or 'softmax_smooth' in self.lossType:
            loss['ce_loss'] = self.ce_loss(outputs[0], labels)
            # loss['ce_loss'] = 0
            # ce_iter = 0
            # for output in outputs[1:]:
            #     loss['ce_loss'] += self.ce_loss(output, labels)
            #     ce_iter += 1
            # loss['ce_loss'] = 2 * loss['ce_loss'] / ce_iter
        if 'triplet' in self.lossType:
            loss['triplet'] = self.triplet_loss(outputs[1], labels)[0]
            # tri_iter = 0
            # for output in outputs[:3]:
            #     loss['triplet'] += self.triplet_loss(output, labels)[0]
            #     tri_iter += 1
            # loss['triplet'] = loss['triplet'] / tri_iter
            # loss['triplet'] = self.triplet_loss(feats, labels)[0]
        # if 'center' in self.lossType:                       loss += 0.0005 * self.center_loss(feats, labels)
        return loss
