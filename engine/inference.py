# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher
from utils.precision_bn import update_bn_stats


@torch.no_grad()
def inference(cfg, model, train_dataloader, test_dataloader, num_query):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    logger.info("compute precise batchnorm ...")
    # model.train()
    # update_bn_stats(model, train_dataloader, num_iters=300)
    model.eval()

    cat_feats, feats, pids, camids = [], [], [], []
    test_prefetcher = data_prefetcher(test_dataloader)
    batch = test_prefetcher.next()
    while batch[0] is not None:
        img, pid, camid = batch
        cat_feat, feat = model(img)
        cat_feats.append(cat_feat.cpu())
        feats.append(feat.cpu())
        pids.extend(np.asarray(pid.cpu().numpy()))
        camids.extend(np.asarray(camid))

        batch = test_prefetcher.next()

    feats = torch.cat(feats, dim=0)
    cat_feats = torch.cat(cat_feats, dim=0)
    if cfg.TEST.NORM:
        feats = F.normalize(feats, p=2, dim=1)
        cat_feats = F.normalize(cat_feats, p=2, dim=1)
    # query
    cat_qf = cat_feats[:num_query]
    qf = feats[:num_query]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    # gallery
    cat_gf = cat_feats[num_query:]
    gf = feats[num_query:]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    # cosine distance
    cat_dist = torch.mm(cat_qf, cat_gf.t())
    distmat = torch.mm(qf, gf.t())

    # IIA post fusion strategy for all query and gallery
    # qf = qf
    # gf = gf
    # m = qf.shape[0]
    # n = gf.shape[0]
    # distmat = torch.zeros((m, n)).to(qf)
    # for i, q_f in enumerate(qf):
    #     print(i)
    #     D = torch.cat([q_f[None, :], gf], dim=0) # [1+g, 2048]
    #     S = torch.mm(D, D.t())  # [1+g, 1+g]
    #     for _ in range(5):
    #         S = S - torch.eye(S.shape[0]).to(S)
    #         s_v, s_i = torch.topk(S, 10, dim=1)
    #         s_v = F.softmax(s_v, dim=1)
    #         s = torch.zeros((S.size()[0], S.size()[0])).to(qf)  # [1+g, 1+g]
    #         for j in range(s_i.shape[0]):
    #             s[j, s_i[j]] = s_v[j]
    #         u = 0.8 * torch.eye(S.size()[0]).to(s) + 0.2 * s
    #         D = torch.mm(u, D)
    #         S = torch.mm(D, D.t())
    #     distmat[i] = S[0][1:]

    cmc, mAP = evaluate(1-distmat.numpy(), q_pids, g_pids, q_camids, g_camids)
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

    cmc, mAP = evaluate(1-cat_dist.numpy(), q_pids, g_pids, q_camids, g_camids)
    logger.info('cat feature')
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

    # IIA post fusion strategy only for query
    # m = qf.shape[0]  # query number
    # n = gf.shape[0]  # gallery number
    # distmat = torch.zeros(m, n)
    # for i, q_f in enumerate(qf):
    #     S = torch.mm(q_f[None, ], gf.t())  # (1, g)
    #     for _ in range(3):
    #         s_v, s_i = torch.topk(S, 3, dim=1)
    #         s_v = F.softmax(s_v, dim=1)
    #         q_f = 0.8 * q_f + 0.2 * torch.mm(s_v, gf[s_i][0])[0]
    #         # s = torch.zeros((1, S.shape[1]))  # (1, g)
    #         # s[0, s_i] = s_v
    #         # q_f = 0.9 * q_f + 0.1 * torch.mm(s, gf)[0]
    #         q_f = F.normalize(q_f, p=2, dim=0)
    #         S = torch.mm(q_f[None], gf.t())
    #     distmat[i] = S[0]
    #
    # euclidean distance
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())

    # distmat = 1 - distmat.numpy()
    # =============================
    # re-rank
    # q_g_dist = distmat
    # q_q_dist = 1 - torch.mm(qf, qf.t()).numpy()
    # g_g_dist = 1 - torch.mm(gf, gf.t()).numpy()
    # distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist, 5, 5)
    # =============================
    # distmat = distmat.numpy()
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    # logger.info(f"mAP: {mAP:.1%}")
    # for r in [1, 5, 10]:
    #     logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

