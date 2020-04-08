# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
import sys
sys.path.append('..')

from fastreid.config import get_cfg
from fastreid.data.transforms import ToTensor
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

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
        "--export-jitmodule",
        action='store_true',
        help="If export reid model to traced jit module"
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class ReidDemo(object):
    """
    ReID demo example
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        if cfg.MODEL.WEIGHTS.endswith('.pt'):
            self.model = torch.jit.load(cfg.MODEL.WEIGHTS)
        else:
            self.model = build_model(cfg)
            # load pre-trained model
            Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

            self.model.eval()
            # self.model = nn.DataParallel(self.model)
            self.model.cuda()

        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, num_channels, 1, 1)
        self.std = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, num_channels, 1, 1)

    def preprocess(self, img):
        img = cv2.resize(img, tuple(self.cfg.INPUT.SIZE_TEST[::-1]))
        img = ToTensor()(img)[None, :, :, :]
        return img.sub_(self.mean).div_(self.std)

    @torch.no_grad()
    def predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = self.preprocess(img)
        output = self.model.inference(data.cuda())
        feat = output.cpu().data.numpy()
        return feat

    @classmethod
    @torch.no_grad()
    def export_jit_model(cls, cfg, model, output_dir):
        example = torch.rand(1, len(cfg.MODEL.PIXEL_MEAN), *cfg.INPUT.SIZE_TEST)
        example = example.cuda()
        # if isinstance(model, (nn.DistributedDataParallel, nn.DataParallel)):
        #     model = model.module
        # else:
        #     model = model
        traced_script_module = torch.jit.trace_module(model, {"inference": example})
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        traced_script_module.save(os.path.join(output_dir, "traced_module.pt"))


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    reidSystem = ReidDemo(cfg)
    if args.export_jitmodule and not isinstance(reidSystem.model, torch.jit.ScriptModule):
        reidSystem.export_jit_model(cfg, reidSystem.model, args.output)

    feats = [reidSystem.predict(data) for data in args.input]

    cos_12 = np.dot(feats[0], feats[1].T).item()
    cos_13 = np.dot(feats[0], feats[2].T).item()
    cos_23 = np.dot(feats[1], feats[2].T).item()

    print('cosine similarity is {:.4f}, {:.4f}, {:.4f}'.format(cos_12, cos_13, cos_23))
