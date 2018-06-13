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
from datasets.samplers import RandomIdentitySampler
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import args
from datasets import data_manager
from datasets.data_loader import ImageData
from models import ResNetBuilder
from trainers import ResNetClsTrainer, ResNetTriTrainer, ResNetClsTriTrainer, ResNetEvaluator
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform


def train_classification(**kwargs):
    args._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(args.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(args._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU {}'.format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    pin_memory = True if use_gpu else False

    tb_writer = SummaryWriter(osp.join(args.save_dir, 'tb_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(args.height, args.width)),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    model = ResNetBuilder(num_classes=dataset.num_train_pids)
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    cls_criterion = nn.CrossEntropyLoss()

    def xent_criterion(cls_scores, targets):
        cls_loss = cls_criterion(cls_scores, targets)
        return cls_loss

    # get optimizer
    optimizer = torch.optim.SGD(
        model.optim_policy(), lr=args.lr, weight_decay=args.weight_decay,
        momentum=args.momentum, nesterov=True
    )

    def adjust_lr(optimizer, ep, decay_ep, gamma):
        decay = gamma ** float(ep // decay_ep)
        for g in optimizer.param_groups:
            g['lr'] = args.lr * decay * g.get('lr_multi', 1)

    start_epoch = args.start_epoch
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # get trainer and evaluator
    reid_trainer = ResNetClsTrainer(model, xent_criterion, tb_writer)
    reid_evaluator = ResNetEvaluator(model)

    # start training
    best_rank1 = -np.inf
    best_epoch = 0
    for epoch in range(start_epoch, args.max_epoch):
        if args.step_size > 0:
            adjust_lr(optimizer, epoch + 1, args.step_size, args.gamma)
        reid_trainer.train(epoch, trainloader, optimizer, args.print_freq)

        # skip if not save model
        if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
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
            }, is_best=is_best, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print(
        'Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def train_triplet(**kwargs):
    args._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(args.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(args._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU {}'.format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    pin_memory = True if use_gpu else False

    tb_writer = SummaryWriter(osp.join(args.save_dir, 'tb_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(args.height, args.width)),
        sampler=RandomIdentitySampler(dataset.train, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    model = ResNetBuilder()
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    tri_criterion = TripletLoss(margin=args.margin)

    def tri_hard(feat, targets):
        tri_loss, _, _ = tri_criterion(feat, targets)
        return tri_loss

    # get optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_ep, gamma):
        if ep < start_decay_ep:
            return
        lr_decay = gamma ** (float(ep - start_decay_ep) /
                             (total_ep - start_decay_ep))
        for g in optimizer.param_groups:
            g['lr'] = base_lr * lr_decay

    start_epoch = args.start_epoch
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # get trainer and evaluator
    reid_trainer = ResNetTriTrainer(model, tri_hard, tb_writer)
    reid_evaluator = ResNetEvaluator(model)

    # start training
    best_rank1 = -np.inf
    best_epoch = 0
    for epoch in range(start_epoch, args.max_epoch):
        if args.step_size > 0:
            adjust_lr_exp(optimizer, args.lr, epoch + 1, args.max_epoch, args.step_size, args.gamma)
        reid_trainer.train(epoch, trainloader, optimizer, args.print_freq)

        # skip if not save model
        if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
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
            }, is_best=is_best, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print(
        'Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def train_cls_triplet(**kwargs):
    args._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(args.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(args._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU {}'.format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    pin_memory = True if use_gpu else False

    tb_writer = SummaryWriter(osp.join(args.save_dir, 'tb_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(args.height, args.width)),
        sampler=RandomIdentitySampler(dataset.train, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    model = ResNetBuilder(num_classes=dataset.num_train_pids)
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    cls_criterion = nn.CrossEntropyLoss()
    tri_criterion = TripletLoss(margin=args.margin)

    def xent_tri_criterion(cls_scores, global_feat, targets):
        cls_loss = cls_criterion(cls_scores, targets)
        tri_loss, dist_ap, dist_an = tri_criterion(global_feat, targets)
        loss = cls_loss + tri_loss
        return loss

    # get optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_ep, gamma):
        if ep < start_decay_ep:
            return
        lr_decay = gamma ** (float(ep - start_decay_ep) /
                             (total_ep - start_decay_ep))
        for g in optimizer.param_groups:
            g['lr'] = base_lr * lr_decay

    start_epoch = args.start_epoch
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # get trainer and evaluator
    reid_trainer = ResNetClsTriTrainer(model, xent_tri_criterion, tb_writer)
    reid_evaluator = ResNetEvaluator(model)

    # start training
    best_rank1 = -np.inf
    best_epoch = 0
    for epoch in range(start_epoch, args.max_epoch):
        if args.step_size > 0:
            adjust_lr_exp(optimizer, args.lr, epoch + 1, args.max_epoch, args.step_size, args.gamma)
        reid_trainer.train(epoch, trainloader, optimizer, args.print_freq)

        # skip if not save model
        if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
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
            }, is_best=is_best, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print(
        'Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def test(**kwargs):
    args._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(args.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

    if use_gpu:
        print('currently using GPU {}'.format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(args.height, args.width)),
        batch_size=args.test_batch, num_workers=args.workers,
        pin_memory=pin_memory
    )

    print('loading model ...')
    model = ResNetBuilder(num_classes=dataset.num_train_pids)
    # ckpt = torch.load(args.load_model)
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
