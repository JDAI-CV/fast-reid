# encoding: utf-8
"""
@author:  l1aoxingyu, guan'an wang
@contact: sherlockliao01@gmail.com, guan.wang0706@gmail.com
"""

from fastreid.config import CfgNode as CN


def add_shufflenet_config(cfg):
    _C = cfg
    _C.MODEL.BACKBONE.MODEL_SIZE = '1.0x'


def add_kdreid_config(cfg):
    _C = cfg

    _C.MODEL_TEACHER = CN()
    _C.MODEL_TEACHER.META_ARCHITECTURE = 'Baseline'

    # ---------------------------------------------------------------------------- #
    # teacher model Backbone options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.BACKBONE = CN()

    _C.MODEL_TEACHER.BACKBONE.NAME = "build_resnet_backbone"
    _C.MODEL_TEACHER.BACKBONE.DEPTH = "50x"
    _C.MODEL_TEACHER.BACKBONE.LAST_STRIDE = 1
    # If use IBN block in backbone
    _C.MODEL_TEACHER.BACKBONE.WITH_IBN = False
    # If use SE block in backbone
    _C.MODEL_TEACHER.BACKBONE.WITH_SE = False
    # If use Non-local block in backbone
    _C.MODEL_TEACHER.BACKBONE.WITH_NL = False

    # for shufflenet
    _C.MODEL_TEACHER.BACKBONE.MODEL_SIZE = '1.0x'

    #
    _C.MODEL_TEACHER.BACKBONE.NORM = 'BN'
    _C.MODEL_TEACHER.BACKBONE.PRETRAIN = False

    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.HEADS = CN()
    _C.MODEL_TEACHER.HEADS.NAME = "BNneckHead"

    # Input feature dimension
    _C.MODEL_TEACHER.HEADS.IN_FEAT = 2048
    # Pooling layer type
    _C.MODEL_TEACHER.HEADS.POOL_LAYER = "avgpool"
    _C.MODEL_TEACHER.HEADS.NECK_FEAT = "before"
    _C.MODEL_TEACHER.HEADS.CLS_LAYER = "linear"

    # Pretrained teacher and student model weights
    _C.MODEL.TEACHER_WEIGHTS = ""
    _C.MODEL.STUDENT_WEIGHTS = ""

    #
    _C.MODEL_TEACHER.HEADS.NORM = 'BN'
    _C.MODEL_TEACHER.HEADS.SCALE = 64
    _C.MODEL_TEACHER.HEADS.MARGIN = 0.35


def update_model_teacher_config(cfg):
    cfg = cfg.clone()

    frozen = cfg.is_frozen()

    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = cfg.MODEL_TEACHER.META_ARCHITECTURE
    # ---------------------------------------------------------------------------- #
    # teacher model Backbone options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.BACKBONE.NAME = cfg.MODEL_TEACHER.BACKBONE.NAME
    cfg.MODEL.BACKBONE.DEPTH = cfg.MODEL_TEACHER.BACKBONE.DEPTH
    cfg.MODEL.BACKBONE.LAST_STRIDE = cfg.MODEL_TEACHER.BACKBONE.LAST_STRIDE
    # If use IBN block in backbone
    cfg.MODEL.BACKBONE.WITH_IBN = cfg.MODEL_TEACHER.BACKBONE.WITH_IBN
    # If use SE block in backbone
    cfg.MODEL.BACKBONE.WITH_SE = cfg.MODEL_TEACHER.BACKBONE.WITH_SE
    # If use Non-local block in backbone
    cfg.MODEL.BACKBONE.WITH_NL = cfg.MODEL_TEACHER.BACKBONE.WITH_NL
    cfg.MODEL.BACKBONE.PRETRAIN = False

    # for shufflenet
    cfg.MODEL.BACKBONE.MODEL_SIZE = cfg.MODEL_TEACHER.BACKBONE.MODEL_SIZE


    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.HEADS.NAME = cfg.MODEL_TEACHER.HEADS.NAME

    # Input feature dimension
    cfg.MODEL.HEADS.IN_FEAT = cfg.MODEL_TEACHER.HEADS.IN_FEAT
    # Pooling layer type
    cfg.MODEL.HEADS.POOL_LAYER = cfg.MODEL_TEACHER.HEADS.POOL_LAYER

    #
    cfg.MODEL.HEADS.SCALE = cfg.MODEL_TEACHER.HEADS.SCALE
    cfg.MODEL.HEADS.MARGIN = cfg.MODEL_TEACHER.HEADS.MARGIN

    if frozen: cfg.freeze()

    return cfg