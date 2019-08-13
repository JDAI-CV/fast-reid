# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F

from data.datasets.eval_reid import evaluate
from fastai.vision import *


@dataclass
class TrackValue(Callback):
    logger: logging.Logger
    total_iter: int

    # def on_batch_end(self, num_batch, last_loss, **kwargs):
    #     if (num_batch+1) % (self.total_iter//3) == 0:
    #         self.logger.info('Iter [{}/{}], loss: {:.4f}'.format(num_batch, self.total_iter, last_loss.item()))

    def on_epoch_end(self, epoch, smooth_loss, **kwargs):
        self.logger.info('Epoch {}[Iter {}], loss: {:.4f}'.format(epoch, self.total_iter, smooth_loss.item()))
            

@dataclass
class LRScheduler(Callback):
    learn: Learner
    lr_sched: Scheduler

    def on_train_begin(self, **kwargs:Any):
        self.opt = self.learn.opt

    def on_epoch_begin(self, **kwargs:Any):
        self.opt.lr = self.lr_sched.step()


class TestModel(LearnerCallback):
    def __init__(self, learn: Learner, test_labels: Iterator, eval_period: int, num_query: int, logger: logging.Logger, norm=True):
        super().__init__(learn)
        self._test_dl = learn.data.test_dl
        self._eval_period = eval_period
        self._norm = norm
        self._logger = logger
        self._num_query = num_query
        pids = []
        camids = []
        for i in test_labels:
            pids.append(i[0])
            camids.append(i[1])
        self.q_pids = np.asarray(pids[:num_query])
        self.q_camids = np.asarray(camids[:num_query])
        self.g_pids = np.asarray(pids[num_query:])
        self.g_camids = np.asarray(camids[num_query:])

    def on_epoch_end(self, epoch, **kwargs: Any):
        # test model performance
        if (epoch + 1) % self._eval_period == 0:
            self._logger.info('Testing ...')
            feats, pids, camids = [], [], []
            self.learn.model.eval()
            with torch.no_grad():
                for imgs, _ in self._test_dl:
                    feat = self.learn.model(imgs)
                    feats.append(feat)
            feats = torch.cat(feats, dim=0)
            if self._norm:
                feats = F.normalize(feats, p=2, dim=1)
            # query
            qf = feats[:self._num_query]
            # gallery
            gf = feats[self._num_query:]
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = to_np(distmat)
            cmc, mAP = evaluate(distmat, self.q_pids, self.g_pids, self.q_camids, self.g_camids)
            self._logger.info("Test Results - Epoch: {}".format(epoch + 1))
            self._logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                self._logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            self.learn.save("model_{}".format(epoch))


class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        from ipdb import set_trace; set_trace()
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
        

class MixUpLoss(Module):
    "Adapt the loss function `crit` to go with mixup."
    
    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'): 
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else: 
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output,target[:,0].long()), self.crit(output,target[:,1].long())
            d = (loss1 * target[:,2] + loss2 * (1-target[:,2])).mean()
        else:  d = self.crit(output, target)
        if self.reduction == 'mean': return d.mean()
        elif self.reduction == 'sum':            return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit


def do_train(
        cfg,
        model,
        data_bunch,
        test_labels,
        opt_func,
        lr_sched,
        loss_func,
        num_query
):
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = Path(cfg.OUTPUT_DIR)
    epochs = cfg.SOLVER.MAX_EPOCHS
    total_iter = len(data_bunch.train_dl)

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start Training")

    cb_fns = [
        partial(LRScheduler, lr_sched=lr_sched),
        partial(TestModel, test_labels=test_labels, eval_period=eval_period, num_query=num_query, logger=logger),
    ]
    if cfg.INPUT.MIXUP:
        cb_fns.append(
            partial(MixUpCallback, alpha=cfg.INPUT.MIXUP_ALPHA))    

    learn = Learner(
        data_bunch,
        model,
        path=output_dir,
        opt_func=opt_func,
        loss_func=loss_func,
        true_wd=False,
        callback_fns=cb_fns,
        callbacks=[TrackValue(logger, total_iter)])

    learn.fit(epochs, lr=cfg.SOLVER.BASE_LR, wd=cfg.SOLVER.WEIGHT_DECAY)
    # learn.recorder.plot_losses()
    # plt.savefig(os.path.join(output_dir, "loss.jpg"))
    # learn.recorder.plot_lr()
    # plt.savefig(os.path.join(output_dir, "lr.jpg"))
