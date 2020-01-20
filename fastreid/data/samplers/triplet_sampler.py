# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

from collections import defaultdict

import random
import copy
import numpy as np
import re
import torch
from torch.utils.data.sampler import Sampler


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


# def No_index(a, b):
#     assert isinstance(a, list)
#     if not isinstance(b, list):
#         return [i for i, j in enumerate(a) if j != b]
#     else:
#         return [i for i, j in enumerate(a) if j not in b]


# class RandomIdentitySampler(Sampler):
#     """Randomly samples N identities each with K instances.
#     Args:
#         data_source (list): contains tuples of (img_path(s), pid, camid).
#         batch_size (int): batch size.
#         num_instances (int): number of instances per identity in a batch.
#     """
#
#     def __init__(self, data_source, batch_size, num_instances):
#         if batch_size < num_instances:
#             raise ValueError(
#                 'batch_size={} must be no less '
#                 'than num_instances={}'.format(batch_size, num_instances)
#             )
#
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.num_instances = num_instances
#         self.num_pids_per_batch = self.batch_size // self.num_instances
#         self.index_dic = defaultdict(list)
#         for index, info in enumerate(self.data_source):
#             pid = info[1]
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#
#         # estimate number of examples in an epoch
#         self.length = 0
#         for pid in self.pids:
#             idxs = self.index_dic[pid]
#             num = len(idxs)
#             if num < self.num_instances:
#                 num = self.num_instances
#             self.length += num - num % self.num_instances
#
#     def __iter__(self):
#         batch_idxs_dict = defaultdict(list)
#
#         for pid in self.pids:
#             idxs = copy.deepcopy(self.index_dic[pid])
#             if len(idxs) < self.num_instances:
#                 idxs = np.random.choice(
#                     idxs, size=self.num_instances, replace=True
#                 )
#             random.shuffle(idxs)
#             batch_idxs = []
#             for idx in idxs:
#                 batch_idxs.append(idx)
#                 if len(batch_idxs) == self.num_instances:
#                     batch_idxs_dict[pid].append(batch_idxs)
#                     batch_idxs = []
#
#         avai_pids = copy.deepcopy(self.pids)
#         final_idxs = []
#
#         while len(avai_pids) >= self.num_pids_per_batch:
#             selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
#             for pid in selected_pids:
#                 batch_idxs = batch_idxs_dict[pid].pop(0)
#                 final_idxs.extend(batch_idxs)
#                 if len(batch_idxs_dict[pid]) == 0:
#                     avai_pids.remove(pid)
#
#         return iter(final_idxs)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, info in enumerate(data_source):
            pid = info[1]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self._seed = 0
        self._shuffle = True

    def __iter__(self):
        indices = self._infinite_indices()
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            yield from t

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self.num_identities, generator=g)
            else:
                yield from torch.arange(self.num_identities)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)

# class RandomIdentitySampler(Sampler):
#     def __init__(self, data_source, batch_size, num_instances):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.num_instances = num_instances
#         self.num_pids_per_batch = self.batch_size // self.num_instances
#         self.index_pid = defaultdict(int)
#         self.index_dic = defaultdict(list)
#         self.pid_cam = defaultdict(list)
#         for index, info in enumerate(data_source):
#             pid = info[1]
#             cam = info[2]
#             self.index_pid[index] = pid
#             self.index_dic[pid].append(index)
#             self.pid_cam[pid].append(cam)
#
#         self.pids = list(self.index_dic.keys())
#         self.num_identities = len(self.pids)
#
#     def __len__(self):
#         return self.num_identities * self.num_instances
#
#     def __iter__(self):
#         indices = torch.randperm(self.num_identities).tolist()
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             all_inds = self.index_dic[pid]
#             chosen_ind = random.choice(all_inds)
#             _, chosen_pid, chosen_cam = self.data_source[chosen_ind]
#             assert chosen_pid == pid, 'id is not matching for self.pids and data_source'
#             tmp_ret = [chosen_ind]
#
#             all_cam = self.pid_cam[pid]
#
#             tmp_cams = [chosen_cam]
#             tmp_inds = [chosen_ind]
#             remain_cam_ind = No_index(all_cam, chosen_cam)
#             ava_inds = No_index(all_inds, chosen_ind)
#             while True:
#                 if remain_cam_ind:
#                     tmp_ind = random.choice(remain_cam_ind)
#                     _, _, tmp_cam = self.data_source[all_inds[tmp_ind]]
#                     tmp_inds.append(tmp_ind)
#                     tmp_cams.append(tmp_cam)
#                     tmp_ret.append(all_inds[tmp_ind])
#                     remain_cam_ind = No_index(all_cam, tmp_cams)
#                     ava_inds = No_index(all_inds, tmp_inds)
#                 elif ava_inds:
#                     tmp_ind = random.choice(ava_inds)
#                     tmp_inds.append(tmp_ind)
#                     tmp_ret.append(all_inds[tmp_ind])
#                     ava_inds = No_index(all_inds, tmp_inds)
#                 else:
#                     tmp_ind = random.choice(all_inds)
#                     tmp_ret.append(tmp_ind)
#
#                 if len(tmp_ret) == self.num_instances:
#                     break
#
#             ret.extend(tmp_ret)
#
#         return iter(ret)
