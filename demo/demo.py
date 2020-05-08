# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('..')

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

cudnn.benchmark = True


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="FastReID demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="traced_module/",
        help="A file or directory to save export jit module.",

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
    demo = DefaultPredictor(cfg)

    feats = []
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = cv2.imread(path)
            feats.append(demo(img))

    cos_12 = np.dot(feats[0], feats[1].T).item()
    cos_13 = np.dot(feats[0], feats[2].T).item()
    cos_23 = np.dot(feats[1], feats[2].T).item()

    print('cosine similarity is {:.4f}, {:.4f}, {:.4f}'.format(cos_12, cos_13, cos_23))
