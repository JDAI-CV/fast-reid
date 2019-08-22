# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from fastai.vision import *
import logging
from data.datasets.eval_reid import evaluate


__all__ = ['TrackValue', 'LRScheduler', 'TestModel', 'CutMix']


class CutMix(LearnerCallback):
    def __init__(self, learn:Learner, cutmix_prob:float=0.5, beta:float=1.0):
        super().__init__(learn)
        self.cutmix_prob,self.beta = cutmix_prob,beta 
    
    @staticmethod
    def rand_bbox(size, lambd):
        h,w = size[2],size[3]
        cut_rat = np.sqrt(1. - lambd)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # Uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        return bbx1, bby1, bbx2, bby2


    def on_batch_begin(self, last_input, last_target, train, epoch, **kwargs):
        if not train: return
        # if epoch > 90:
            # lambd = torch.ones(last_target.size(0)).to(last_input.device)
            # new_target = torch.cat([last_target[:, None].float(), last_target[:, None].float(), lambd[:,None].float()], 1) 
            # return {'last_target': new_target}
        if np.random.rand(1) > self.cutmix_prob: return
        lambd = np.random.beta(self.beta, self.beta)
        lambd = max(lambd, 1-lambd)
        # lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(last_input.size(), lambd)
        last_input[:, :, bby1:bby2, bbx1:bbx2] = x1[:, :, bby1:bby2, bbx1:bbx2]
        # Adjust lambda to exactly match pixel ratio
        lambd = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (last_input.size()[-1] * last_input.size()[-2]))
        lambd = torch.ones(last_target[:,None].size(), dtype=torch.float32).fill_(lambd).to(last_input.device)
        new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd], 1)
        return {'last_input': last_input, 'last_target': new_target}


@dataclass
class TrackValue(Callback):
    logger: logging.Logger
    total_iter: int

    def on_epoch_end(self, epoch, smooth_loss, **kwargs):
        self.logger.info(f'Epoch {epoch}[Iter {self.total_iter}], loss: {smooth_loss.item():.4f}')
            

class LRScheduler(LearnerCallback):
    def __init__(self, learn, lr_sched):
        super().__init__(learn)
        self.lr_sched = lr_sched

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
            self._logger.info(f"Test Results - Epoch: {epoch+1}")
            self._logger.info(f"mAP: {mAP:.1%}")
            for r in [1, 5, 10]:
                self._logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r-1]:.1%}")
            self.learn.save("model_{}".format(epoch))