# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

import torch
sys.path.append('../..')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup
from fastreid.modeling.meta_arch import build_model
from fastreid.export.tensorflow_export import export_tf_reid_model
from fastreid.export.tf_modeling import TfMetaArch


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = 50
    cfg.MODEL.BACKBONE.LAST_STRIDE = 1
    # If use IBN block in backbone
    cfg.MODEL.BACKBONE.WITH_IBN = False
    cfg.MODEL.BACKBONE.PRETRAIN = False

    from torchvision.models import resnet50
    # model = TfMetaArch(cfg)
    model = resnet50(pretrained=False)
    # model.load_params_wo_fc(torch.load('logs/bjstation/res50_baseline_v0.4/ckpts/model_epoch80.pth'))
    model.eval()
    dummy_inputs = torch.randn(1, 3, 256, 128)
    export_tf_reid_model(model, dummy_inputs, 'reid_tf.pb')
