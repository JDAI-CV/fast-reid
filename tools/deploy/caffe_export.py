# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

import argparse

import torch
import sys
sys.path.append('../../')

import pytorch_to_caffe
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.file_io import PathManager
from fastreid.utils.checkpoint import Checkpointer


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Convert Pytorch to Caffe model")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='caffe_model',
        help='path to save converted caffe model'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.POOL_LAYER = "identity"
    cfg.MODEL.BACKBONE.WITH_NL = False

    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.cuda()
    model.eval()
    print(model)

    inputs = torch.randn(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]).cuda()
    PathManager.mkdirs(args.output)
    pytorch_to_caffe.trans_net(model, inputs, args.name)
    pytorch_to_caffe.save_prototxt(f"{args.output}/{args.name}.prototxt")
    pytorch_to_caffe.save_caffemodel(f"{args.output}/{args.name}.caffemodel")
