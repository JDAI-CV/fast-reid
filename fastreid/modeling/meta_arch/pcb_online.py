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
        if self.training:
            targets = batched_inputs['targets']
            losses = self.losses(outputs, targets)
            return losses
        else:
            return outputs

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        pred_query_feature       = outputs['query_feature']
        pred_gallery_feature     = outputs['gallery_feature']
        pred_class_logits        = outputs['pred_class_logits'].detach()
        cls_outputs              = outputs['cls_outputs']
        
        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'ContrastiveLoss' in loss_names:
            contrastive_kwargs = self.loss_kwargs.get('contrastive')
            loss_dict['loss_contrastive'] = contrastive_loss(
                pred_query_feature,
				pred_gallery_feature,
                gt_labels,
                contrastive_kwargs.get('margin')
            ) * contrastive_kwargs.get('scale')

        return loss_dict
