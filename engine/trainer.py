# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from data.datasets.eval_reid import evaluate
from fastai.vision import *


class LrScheduler(LearnerCallback):
    def __init__(self, learn: Learner, lr_sched: Scheduler):
        super().__init__(learn)
        self.lr_sched = lr_sched

    def on_train_begin(self, **kwargs: Any) -> None:
        self.opt = self.learn.opt

    def on_epoch_begin(self, **kwargs: Any) -> None:
        self.opt.lr = self.lr_sched.step()


class TestModel(LearnerCallback):
    def __init__(self, learn: Learner, test_labels: Iterator, eval_period: int, num_query: int, output_dir: Path):
        super().__init__(learn)
        self.test_dl = learn.data.test_dl
        self.eval_period = eval_period
        self.output_dir = output_dir
        self.num_query = num_query
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
        if (epoch + 1) % self.eval_period == 0:
            print('Testing ...')
            feats, pids, camids = [], [], []
            self.learn.model.eval()
            with torch.no_grad():
                for imgs, _ in self.test_dl:
                    feat = self.learn.model(imgs)
                    feats.append(feat)

            feats = torch.cat(feats, dim=0)
            # query
            qf = feats[:self.num_query]
            # gallery
            gf = feats[self.num_query:]
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
            self.learn.save(self.output_dir / 'reid_model_{}'.format(epoch))


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
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS

    print("Start training")

    learn = Learner(data_bunch, model, opt_func=opt_func, loss_func=loss_func, true_wd=False)

    lr_sched_cb = LrScheduler(learn, lr_sched)
    testmodel_cb = TestModel(learn, test_labels, eval_period, num_query, Path(output_dir))

    learn.fit(epochs, callbacks=[lr_sched_cb, testmodel_cb],
              lr=cfg.SOLVER.BASE_LR, wd=cfg.SOLVER.WEIGHT_DECAY)
