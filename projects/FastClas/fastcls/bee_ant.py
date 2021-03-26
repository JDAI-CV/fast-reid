# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["Hymenoptera"]


@DATASET_REGISTRY.register()
class Hymenoptera(ImageDataset):
    """This is a demo dataset for smoke test, you can refer to
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    dataset_dir = 'hymenoptera_data'
    dataset_name = "hyt"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")

        required_files = [
            self.dataset_dir,
            train_dir,
            val_dir,
        ]
        self.check_before_run(required_files)

        self.classes, self.class_to_idx = self._find_classes(train_dir)

        train = self.process_dir(train_dir)
        val = self.process_dir(val_dir)

        super().__init__(train, val, [], **kwargs)

    def process_dir(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, self.class_to_idx[dir_name], '0'])
        return data

    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
