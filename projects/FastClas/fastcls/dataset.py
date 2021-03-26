# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image


class ClasDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform

        pid_set = set()
        for i in img_items:
            pid_set.add(i[1])

        self.pids = sorted(list(pid_set))

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)

        return {
            "images": img,
            "targets": pid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)
