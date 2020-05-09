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
from predictor import FeatureExtractionDemo

cudnn.benchmark = True


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--device',
        default='cuda: 1',
        help='CUDA device to use'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
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
    demo = FeatureExtractionDemo(cfg, device=args.device, parallel=args.parallel)

    feats = []
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            img = cv2.imread(path)
            feat = demo.run_on_image(img)
            feats.append(feat.numpy())

    cos_sim = np.dot(feats[0], feats[1].T).item()

    print('cosine similarity of the first two images is {:.4f}'.format(cos_sim))
