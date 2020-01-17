# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
sys.path.append('.')
from data import get_dataloader
from config import cfg
import argparse
from data.datasets import init_dataset
# cfg.DATALOADER.SAMPLER = 'triplet'
cfg.DATASETS.NAMES = ("market1501", "dukemtmc", "cuhk03", "msmt17",)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        '-cfg', "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    # dataset = init_dataset('msmt17', combineall=True)
    get_dataloader(cfg)
    # tng_dataloader, val_dataloader, num_classes, num_query = get_dataloader(cfg)
    # def get_ex(): return open_image('datasets/beijingStation/query/000245_c10s2_1561732033722.000000.jpg')
    # im = get_ex()
    # print(data.train_ds[0])
    # print(data.test_ds[0])
    # a = next(iter(data.train_dl))
    # from IPython import embed; embed()
    # from ipdb import set_trace; set_trace()
    # im.apply_tfms(crop_pad(size=(300, 300)))
