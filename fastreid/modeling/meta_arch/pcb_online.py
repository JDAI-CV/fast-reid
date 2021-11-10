# coding: utf-8
"""
Sun, Y. ,  Zheng, L. ,  Yang, Y. ,  Tian, Q. , &  Wang, S. . (2017). Beyond part models: person retrieval with refined part pooling (and a strong convolutional baseline). Springer, Cham.
实现和线上一模一样的PCB
"""
import torch
import torch.nn.functional as F

from fastreid.modeling.losses import cross_entropy_loss, log_accuracy, contrastive_loss
from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class PcbOnline(Baseline):

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        bsz = int(images.size(0) / 2)
        feats = self.backbone(images)
        feats = torch.cat((feats['full'], feats['parts'][0], feats['parts'][1], feats['parts'][2]), 1)
        feats = F.normalize(feats, p=2.0, dim=-1)

        qf = feats[0: bsz * 2: 2, ...]
        xf = feats[1: bsz * 2: 2, ...]
        outputs = self.heads({'query': qf, 'gallery': xf})

        outputs['query_feature'] = qf
        outputs['gallery_feature'] = xf
        outputs['features'] = {}
        if self.training:
            targets = batched_inputs['targets']
            losses = self.losses(outputs, targets)
            return losses
        else:
            return outputs

