# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from data.datasets.eval_reid import evaluate
from fastai.vision import *


@dataclass
class TrackValue(Callback):
    log_path: Path
    total_iter: int

    def on_batch_end(self, num_batch, last_loss, **kwargs:Any):
        if (num_batch+1) % (self.total_iter//3) == 0:
            with self.log_path.open('a') as f:
                f.write('Iter [{}/{}], loss: {:.4f}\n'.format(num_batch, self.total_iter, last_loss.item()))

    def on_epoch_end(self, epoch, smooth_loss, **kwargs:Any):
        with self.log_path.open('a') as f:
            f.write('Epoch {}, loss: {:.4f}\n\n'.format(epoch, smooth_loss.item()))


@dataclass
class LRScheduler(Callback):
    learn: Learner
    lr_sched: Scheduler

    def on_train_begin(self, **kwargs:Any):
        self.opt = self.learn.opt

    def on_epoch_begin(self, **kwargs:Any):
        self.opt.lr = self.lr_sched.step()


class TestModel(LearnerCallback):
    def __init__(self, learn: Learner, test_labels: Iterator, eval_period: int, num_query: int,
                 output_dir: Path, log_path: Path, norm=True):
        super().__init__(learn)
        self._test_dl = learn.data.test_dl
        self._eval_period = eval_period
        self._norm = norm
        self._output_dir = output_dir
        self._log_path = log_path
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

    def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        # test model performance
        if (epoch + 1) % self._eval_period == 0:
            print('Testing ...')
            feats, pids, camids = [], [], []
            self.learn.model.eval()
            with torch.no_grad():
                for imgs, _ in self._test_dl:
                    feat = self.learn.model(imgs)
                    feats.append(feat)

            if self._norm:
                feats = torch.norm(feats, p=2, dim=1)
            feats = torch.cat(feats, dim=0)
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
            print("Test Results - Epoch: {}".format(epoch + 1))
            print("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            self.learn.save(self._output_dir / 'reid_model_{}'.format(epoch))
            with open(self.log_path, 'a') as f:
                f.write("Test Results - Epoch: {}\n mAP: {:.1%}\n".format(epoch + 1, mAP))
                for r in [1, 5, 10]:
                    f.write("CMC curve, Rank-{:<3}:{:.1%}\n".format(r, cmc[r - 1]))
                f.write('\n')




def do_train(
        cfg,
        log_path,
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

    print("Start training")

    learn = Learner(
        data_bunch,
        model,
        opt_func=opt_func,
        loss_func=loss_func,
        true_wd=False,
        callback_fns=[
            partial(LRScheduler, lr_sched=lr_sched),
            partial(TestModel,test_labels=test_labels, eval_period=eval_period,
                    num_query=num_query, output_dir=output_dir, log_path=log_path)],
        callbacks=[TrackValue(log_path, total_iter)])

    learn.fit(epochs, lr=cfg.SOLVER.BASE_LR, wd=cfg.SOLVER.WEIGHT_DECAY)
