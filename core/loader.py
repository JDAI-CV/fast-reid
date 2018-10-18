from __future__ import print_function, absolute_import

from collections import defaultdict

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader

from utils import augmenter
from .data_manager import init_dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = np.random.permutation(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


def get_data_provider(opt):
    num_gpus = (len(opt.network.gpus) + 1) // 2
    test_batch_size = opt.test.batch_size * num_gpus

    # data augmenter
    random_mirror = opt.aug.get('random_mirror', False)
    pad = opt.aug.get('pad', False)
    random_crop = opt.aug.get('random_crop', False)
    random_erasing = opt.aug.get('random_erasing', False)

    h, w = opt.aug.resize_size
    train_aug = list()
    train_aug.append(T.Resize((h, w)))
    if random_mirror:
        train_aug.append(T.RandomHorizontalFlip())
    if pad:
        train_aug.append(T.Pad(padding=pad))
    if random_crop:
        train_aug.append(T.RandomCrop((h, w)))
    train_aug.append(T.ToTensor())
    train_aug.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if random_erasing:
        train_aug.append(augmenter.RandomErasing())
    train_aug = T.Compose(train_aug)

    test_aug = list()
    test_aug.append(T.Resize((h, w)))
    test_aug.append(T.ToTensor())
    test_aug.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_aug = T.Compose(test_aug)

    dataset = init_dataset(opt.dataset.name)
    train_set = ImageData(dataset.train, train_aug)
    test_set = ImageData(dataset.query + dataset.gallery, test_aug)

    if opt.train.sampler == 'softmax':
        train_batch_size = opt.train.batch_size * num_gpus
        train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True,
                                  num_workers=opt.network.workers, pin_memory=True, drop_last=True)
    elif opt.train.sampler == 'triplet':
        train_batch_size = opt.train.p_size * num_gpus * opt.train.k_size
        train_loader = DataLoader(train_set, batch_size=train_batch_size,
                                  sampler=RandomIdentitySampler(dataset.train, opt.train.k_size),
                                  num_workers=opt.network.workers, pin_memory=True)
    else:
        raise ValueError('sampler must be softmax or triplet, but get {}'.format(opt.train.sampler))

    test_loader = DataLoader(test_set, batch_size=test_batch_size, num_workers=opt.network.workers, pin_memory=True)
    return train_loader, test_loader, len(dataset.query)  # return number of query


if __name__ == "__main__":
    from config import opt

    train_loader, test_loader, num_query = get_data_provider(opt)
    from IPython import embed

    embed()
