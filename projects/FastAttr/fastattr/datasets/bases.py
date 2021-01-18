# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import logging
import os

from tabulate import tabulate
from termcolor import colored

logger = logging.getLogger("fastreid.attr_dataset")


class Dataset(object):

    def __init__(
            self,
            train,
            val,
            test,
            attr_dict,
            mode='train',
            verbose=True,
            **kwargs,
    ):
        self.train = train
        self.val = val
        self.test = test
        self._attr_dict = attr_dict
        self._num_attrs = len(self.attr_dict)

        if mode == 'train':
            self.data = self.train
        elif mode == 'val':
            self.data = self.val
        else:
            self.data = self.test

    @property
    def num_attrs(self):
        return self._num_attrs

    @property
    def attr_dict(self):
        return self._attr_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def combine_all(self):
        """Combines train, val and test in a dataset for training."""
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def show_train(self):
        num_train = len(self.train)
        num_val = len(self.val)
        num_total = num_train + num_val

        headers = ['subset', '# images']
        csv_results = [
            ['train', num_train],
            ['val', num_val],
            ['total', num_total],
        ]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
        logger.info("attributes:")
        for label, attr in self.attr_dict.items():
            logger.info('{:3d}: {}'.format(label, attr))
        logger.info("------------------------------")
        logger.info("# attributes: {}".format(len(self.attr_dict)))

    def show_test(self):
        num_test = len(self.test)

        headers = ['subset', '# images']
        csv_results = [
            ['test', num_test],
        ]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
