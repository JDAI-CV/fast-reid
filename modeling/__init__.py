# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .mask_model import Maskmodel
from .ms_baseline import MSBaseline


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'baseline':
        model = Baseline(cfg.MODEL.BACKBONE,
                         num_classes,
                         cfg.MODEL.LAST_STRIDE,
                         cfg.MODEL.WITH_IBN,
                         cfg.MODEL.WITH_SE,
                         cfg.MODEL.GCB,
                         cfg.MODEL.STAGE_WITH_GCB,
                         cfg.MODEL.PRETRAIN,
                         cfg.MODEL.PRETRAIN_PATH)
    elif cfg.MODEL.NAME == 'maskmodel':
        model = Maskmodel(cfg.MODEL.BACKBONE,
                          num_classes,
                          cfg.MODEL.LAST_STRIDE,
                          cfg.MODEL.WITH_IBN,
                          cfg.MODEL.WITH_SE,
                          cfg.MODEL.GCB,
                          cfg.MODEL.STAGE_WITH_GCB,
                          cfg.MODEL.PRETRAIN,
                          cfg.MODEL.PRETRAIN_PATH)
    elif cfg.MODEL.NAME == 'msbaseline':
        model = MSBaseline(cfg.MODEL.BACKBONE,
                           num_classes,
                           cfg.MODEL.LAST_STRIDE,
                           cfg.MODEL.WITH_IBN,
                           cfg.MODEL.WITH_SE,
                           cfg.MODEL.GCB,
                           cfg.MODEL.STAGE_WITH_GCB,
                           cfg.MODEL.PRETRAIN,
                           cfg.MODEL.PRETRAIN_PATH)
    else:
        raise NameError(f'not support {cfg.MODEL.NAME}')
    return model
