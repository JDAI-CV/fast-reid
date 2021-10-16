# encoding: utf-8

import torch
from torch import nn

from fastreid.modeling.losses import triplet_loss, pairwise_circleloss, pairwise_cosface, contrastive_loss
from .baseline import Baseline
from .build import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class Metric(Baseline):

    def losses(self, outputs, gt_labels):
        pred_features     = outputs['features']

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        if 'ContrastiveLoss' in loss_names:
            contrastive_kwargs = self.loss_kwargs.get('contrastive')
            loss_dict['loss_contrastive'] = contrastive_loss(
                pred_features,
                gt_labels,
                contrastive_kwargs.get('margin')
            ) * contrastive_kwargs.get('scale')

        return loss_dict
