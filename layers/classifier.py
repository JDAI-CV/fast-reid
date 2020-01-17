# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn
from modeling.losses import *
from modeling.backbones import *
from .batch_norm import bn_no_bias
from modeling.utils import *


class ClassBlock(nn.Module):
    """
    Define the bottleneck and classifier layer
    |--bn--|--relu--|--linear--|--classifier--|
    """
    def __init__(self, in_features, num_classes, relu=True, num_bottleneck=512, fc_layer='softmax'):
        super().__init__()
        block1 = []
        block1 += [nn.Linear(in_features, num_bottleneck, bias=False)]
        block1 += [nn.BatchNorm1d(in_features)]
        if relu:
            block1 += [nn.LeakyReLU(0.1)]
        self.block1 = nn.Sequential(*block1)

        self.bnneck = bn_no_bias(num_bottleneck)

        if fc_layer == 'softmax':
            self.classifier = nn.Linear(num_bottleneck, num_classes, bias=False)
        elif fc_layer == 'circle_loss':
            self.classifier = CircleLoss(num_bottleneck, num_classes, s=256, m=0.25)

    def init_parameters(self):
        self.block1.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        x = self.block1(x)
        x = self.bnneck(x)
        if self.training:
            # cls_out = self.classifier(x, label)
            cls_out = self.classifier(x)
            return cls_out
        else:
            return x
