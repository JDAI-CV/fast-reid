from fastreid.modeling import losses as Loss
from torch import nn
import torch
import pdb
def reid_losses(cfg, pred_class_logits, global_features, gt_classes, prefix='') -> dict:
    loss_dict = {}
    if pred_class_logits is not None:
        loss = getattr(Loss, cfg.MODEL.LOSSES.NAME[0])(cfg)(pred_class_logits, global_features, gt_classes)
        loss_dict.update(loss)
    if global_features is not None:
        loss = getattr(Loss, cfg.MODEL.LOSSES.NAME[1])(cfg)(pred_class_logits, global_features, gt_classes)
        loss_dict.update(loss)
    # rename
    named_loss_dict = {}
    for name in loss_dict.keys():
        if name == 'loss_cls':
            named_loss_dict[prefix + name] = loss_dict[name]
        if name == 'loss_triplet':
            named_loss_dict[prefix + name] = loss_dict[name]
    del loss_dict
    return named_loss_dict