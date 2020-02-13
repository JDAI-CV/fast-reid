# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.modeling.model_utils import weights_init_kaiming, weights_init_classifier
from fastreid.modeling.heads import REID_HEADS_REGISTRY


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


@REID_HEADS_REGISTRY.register()
class GeM_BN_Linear(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.gem_pool = GeneralizedMeanPoolingP()
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self._num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        global_features = self.gem_pool(features)
        global_features = global_features.view(global_features.shape[0], -1)
        bn_features = self.bnneck(global_features)

        if not self.training:
            return F.normalize(bn_features),

        pred_class_logits = self.classifier(bn_features)
        return pred_class_logits, global_features, targets,
