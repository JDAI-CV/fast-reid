# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .bce_loss import cross_entropy_sigmoid_loss


@META_ARCH_REGISTRY.register()
class AttrBaseline(Baseline):

    def losses(self, outs, sample_weight=None):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        # pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        # fmt: on

        # Log prediction accuracy
        # log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "BinaryCrossEntropyLoss" in loss_names:
            loss_dict['loss_bce'] = cross_entropy_sigmoid_loss(
                cls_outputs,
                gt_labels,
                sample_weight,
            ) * self._cfg.MODEL.LOSSES.BCE.SCALE

        return loss_dict
