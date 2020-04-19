# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

from .. import losses as Loss


def reid_losses(cfg, pred_class_logits, global_features, gt_classes, prefix='') -> dict:
    loss_dict = {}
    for loss_name in cfg.MODEL.LOSSES.NAME:
        loss = getattr(Loss, loss_name)(cfg)(pred_class_logits, global_features, gt_classes)
        loss_dict.update(loss)
    # rename
    named_loss_dict = {}
    for name in loss_dict.keys():
        named_loss_dict[prefix + name] = loss_dict[name]
    del loss_dict
    return named_loss_dict
