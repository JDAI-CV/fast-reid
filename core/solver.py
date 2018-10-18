# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time

import numpy as np
import torch

from utils.meters import AverageMeter
from utils.serialization import save_checkpoint


class Solver(object):
    def __init__(self, opt, net):
        self.opt = opt
        self.net = net
        self.loss = AverageMeter('loss')
        self.acc = AverageMeter('acc')

    def fit(self, train_data, test_data, num_query, optimizer, criterion, lr_scheduler):
        best_rank1 = -np.inf
        for epoch in range(self.opt.train.num_epochs):
            self.loss.reset()
            self.acc.reset()
            self.net.train()
            # update learning rate
            lr = lr_scheduler.update(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logging.info('Epoch [{}] learning rate update to {:.3e}'.format(epoch, lr))

            tic = time.time()
            btic = time.time()
            for i, inputs in enumerate(train_data):
                data, pids, _ = inputs
                label = pids.cuda()
                score, feat = self.net(data)
                loss = criterion(score, feat, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss.update(loss.item())
                acc = (score.max(1)[1] == label.long()).float().mean().item()
                self.acc.update(acc)

                log_interval = self.opt.misc.log_interval
                if log_interval and not (i + 1) % log_interval:
                    loss_name, loss_value = self.loss.get()
                    metric_name, metric_value = self.acc.get()
                    logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\t'
                                 '%s=%f' % (
                                     epoch, i + 1, train_data.batch_size * log_interval / (time.time() - btic),
                                     loss_name, loss_value,
                                     metric_name, metric_value
                                 ))
                    btic = time.time()

            loss_name, loss_value = self.loss.get()
            metric_name, metric_value = self.acc.get()
            throughput = int(train_data.batch_size * len(train_data) / (time.time() - tic))

            logging.info('[Epoch %d] training: %s=%f\t%s=%f' % (
                epoch, loss_name, loss_value, metric_name, metric_value))
            logging.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))

            is_best = False
            if test_data is not None and self.opt.misc.eval_step and not (epoch + 1) % self.opt.misc.eval_step:
                rank1 = self.test_func(test_data, num_query)
                is_best = rank1 > best_rank1
                if is_best:
                    best_rank1 = rank1
            state_dict = self.net.module.state_dict()
            if not (epoch + 1) % self.opt.misc.save_step:
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': epoch + 1,
                }, is_best=is_best, save_dir=self.opt.misc.save_dir,
                    filename=self.opt.network.name + '.pth.tar')

    def test_func(self, test_data, num_query):
        self.net.eval()
        feat, person, camera = list(), list(), list()
        for inputs in test_data:
            data, pids, camids = inputs
            with torch.no_grad():
                outputs = self.net(data).cpu()
            feat.append(outputs)
            person.extend(pids.numpy())
            camera.extend(camids.numpy())
        feat = torch.cat(feat, 0)
        qf = feat[:num_query]
        q_pids = np.asarray(person[:num_query])
        q_camids = np.asarray(camera[:num_query])
        gf = feat[num_query:]
        g_pids = np.asarray(person[num_query:])
        g_camids = np.asarray(camera[num_query:])

        logging.info("Extracted features for query set, obtained {}-by-{} matrix".format(
            qf.shape[0], qf.shape[1]))
        logging.info("Extracted features for gallery set, obtained {}-by-{} matrix".format(
            gf.shape[0], gf.shape[1]))

        logging.info("Computing distance matrix")

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        logging.info("Computing CMC and mAP")
        cmc, mAP = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in [1, 5, 10]:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        return cmc[0]

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP
