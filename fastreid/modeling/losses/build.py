# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for loss, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
It must returns an instance of :class:`Loss`.
"""


def build_criterion(cfg):
    """
    Build a loss from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Loss`
    """

    loss_names = cfg.MODEL.LOSSES.NAME
    loss_funcs = [LOSS_REGISTRY.get(loss_name)(cfg) for loss_name in loss_names]

    def criterion(*args):
        loss_dict = {}
        for loss_func in loss_funcs:
            loss = loss_func(*args)
            loss_dict.update(loss)
        return loss_dict
    return criterion
