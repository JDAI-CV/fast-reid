# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

from torch.backends import cudnn

sys.path.append(".")
from config import cfg
from utils.logger import setup_logger
from engine.trainer import ReidSystem
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description="ReID Model Training")
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
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # cfg.SOLVER.DIST = num_gpus > 1

    # if cfg.SOLVER.DIST:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )
    #     torch.cuda.synchronize()

    cfg.freeze()

    log_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST_NAMES, cfg.MODEL.VERSION)
    if not os.path.exists(log_save_dir): os.makedirs(log_save_dir)

    logger = setup_logger("reid_baseline.train", log_save_dir, 0)
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info('start training')
    cudnn.benchmark = True

    writer = SummaryWriter(os.path.join(log_save_dir, 'tf'))
    reid_system = ReidSystem(cfg, logger, writer)
    reid_system.train()

    # TODO: continue training
    # if cfg.MODEL.CHECKPOINT is not '':
    #     state = torch.load(cfg.MODEL.CHECKPOINT)
    #     if set(state.keys()) == {'model', 'opt'}:
    #         model_state = state['model']
    #         learn.model.load_state_dict(model_state)
    #         learn.create_opt(0, 0)
    #         learn.opt.load_state_dict(state['opt'])
    #     else:
    #         learn.model.load_state_dict(state['model'])
    #     logger.info(f'continue training from checkpoint {cfg.MODEL.CHECKPOINT}')


if __name__ == '__main__':
    main()
