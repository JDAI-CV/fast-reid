# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models import get_baseline_model
from trainers import clsTrainer, cls_tripletTrainer, tripletTrainer, ResNetEvaluator
from utils.loss import TripletLoss, CrossEntropyLabelSmooth
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform


def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset)

    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    if 'triplet' in opt.model_name:
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.height, opt.width)),
            sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True
        )
    else:
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.height, opt.width)),
            batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
            pin_memory=pin_memory
        )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    if opt.model_name == 'softmax' or opt.model_name == 'softmax_triplet':
        model, optim_policy = get_baseline_model(dataset.num_train_pids)
    elif opt.model_name == 'triplet':
        model, optim_policy = get_baseline_model(num_classes=None)
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    # xent_criterion = nn.CrossEntropyLoss()
    xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids)
    tri_criterion = TripletLoss(opt.margin)

    def cls_criterion(cls_scores, targets):
        cls_loss = xent_criterion(cls_scores, targets)
        return cls_loss

    def triplet_criterion(feat, targets):
        triplet_loss, _, _ = tri_criterion(feat, targets)
        return triplet_loss

    def cls_tri_criterion(cls_scores, feat, targets):
        cls_loss = xent_criterion(cls_scores, targets)
        triplet_loss, _, _ = tri_criterion(feat, targets)
        loss = cls_loss + triplet_loss
        return loss

    # get optimizer
    optimizer = torch.optim.Adam(
        optim_policy, lr=opt.lr, weight_decay=opt.weight_decay
    )

    def adjust_lr(optimizer, ep):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            lr = 1e-3 * opt.num_gpu
        elif ep < 180:
            lr = 1e-4 * opt.num_gpu
        elif ep < 300:
            lr = 1e-5 * opt.num_gpu
        elif ep < 320:
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80) * opt.num_gpu
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            lr = 1e-4 * opt.num_gpu
        else:
            lr = 1e-5 * opt.num_gpu
        for p in optimizer.param_groups:
            p['lr'] = lr

    start_epoch = opt.start_epoch
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # get trainer and evaluator
    if opt.model_name == 'softmax':
        reid_trainer = clsTrainer(opt, model, optimizer, cls_criterion, summary_writer)
    elif opt.model_name == 'softmax_triplet':
        reid_trainer = cls_tripletTrainer(opt, model, optimizer, cls_tri_criterion, summary_writer)
    elif opt.model_name == 'triplet':
        reid_trainer = tripletTrainer(opt, model, optimizer, triplet_criterion, summary_writer)
    reid_evaluator = ResNetEvaluator(model)

    # start training
    best_rank1 = -np.inf
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.step_size > 0:
            adjust_lr(optimizer, epoch + 1)
        reid_trainer.train(epoch, trainloader)

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            rank1 = reid_evaluator.evaluate(queryloader, galleryloader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=is_best, save_dir=opt.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print(
        'Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def test(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    if use_gpu:
        print('currently using GPU {}'.format(opt.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset)

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('loading model ...')
    model, optim_policy = get_baseline_model(dataset.num_train_pids)
    # ckpt = torch.load(opt.load_model)
    # model.load_state_dict(ckpt['state_dict'])
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    reid_evaluator = ResNetEvaluator(model)
    reid_evaluator.evaluate(queryloader, galleryloader)


if __name__ == '__main__':
    import fire

    fire.Fire()
