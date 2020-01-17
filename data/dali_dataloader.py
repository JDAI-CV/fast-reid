# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import random

import torch
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import sys
sys.path.append('./')

from data.datasets import init_dataset

logger = logging.getLogger(__name__)

times = 0


# ref: https://github.com/hszhao/semseg/blob/5e5a0ba7a1fa2cc06f3e8c060cbedff08e160d33/util/dataset.py#L17
def load_and_check_flist(split='train', data_root=None, data_list=None):
    """ Load and check the input filelist
    """
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    logger.info("Totally {} samples in {} set.".format(len(list_read), split))
    logger.info("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split()  # TODO: if split char is '\t' may cause bug
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])

        item = (image_name, label_name)
        image_label_list.append(item)
    logger.info("Checking image&label pair {} list done!".format(split))
    return image_label_list


class FileListIterator(object):
    """ produce the files according to a given file list iteratively.  """

    def __init__(self, file_list, batch_size, split='train'):
        self.data_list = file_list
        self.batch_size = batch_size
        self.split = split
        # self.data_list = load_and_check_flist(split, data_root, file_list)

        self.i = 0
        self.n = len(self.data_list)
        if self.split == 'train':
            random.shuffle(self.data_list)

    @property
    def size(self):
        return self.n

    def __iter__(self, ):
        self.i = 0
        if self.split == 'train':
            random.shuffle(self.data_list)
        return self

    def __next__(self):
        try:
            batch = []
            labels = []
            camids = []
            for _ in range(self.batch_size):
                source_path, target, camid = self.data_list[self.i]

                source = np.frombuffer(open(source_path, 'rb').read(), dtype=np.uint8)
                # target = np.frombuffer(open(target_path, 'rb').read(), dtype=np.uint8)

                batch.append(source)
                # labels.append(target)
                labels.append(np.array([target], dtype=np.uint8))
                camids.append(np.array([camid], dtype=np.uint8))
                self.i = (self.i + 1)
        except:
            raise StopIteration

        return (batch, labels, camids,)

    next = __next__


class ReidPipeline(Pipeline):
    def __init__(
            self, file_list, batch_size,
            num_threads=4, device_id=1, split='train'
    ):
        super().__init__(
            batch_size, num_threads, device_id)

        self.dataset = FileListIterator(
            file_list, batch_size)

        self.source_feeder = ops.ExternalSource()
        self.target_feeder = ops.ExternalSource()
        self.camid_feeder = ops.ExternalSource()

        self.source_decoder = ops.ImageDecoder(
            device='mixed', output_type=types.RGB)
        # self.target_decoder = ops.ImageDecoder(
        #     device='mixed', output_type=types.GRAY)

        self.resize = ops.Resize(device='gpu', resize_x=128, resize_y=384)
        self.source_convas = ops.Paste(device='gpu', fill_value=(0, 0, 0), ratio=1.05, min_canvas_size=148)

        # self.source_convas = ops.Paste(device='gpu', fill_value=(125, 128, 127), ratio=1.0,
        #                                min_canvas_size=crop_size[0], )
        # self.target_convas = ops.Paste(device='gpu', fill_value=(255,), ratio=1.0, min_canvas_size=crop_size[0], )
        self.pos_x_rng = ops.Uniform(range=(0, 1))
        self.pos_y_rng = ops.Uniform(range=(0, 1))

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            crop=(384, 128),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.mirror_rng = ops.CoinFlip(probability=0.5)

        self.iterator = iter(self.dataset)

    def define_graph(self, ):
        self.source = self.source_feeder()
        self.target = self.target_feeder()
        self.camid = self.camid_feeder()
        image = self.source_decoder(self.source)
        # label = self.target_decoder(self.target)

        # # Apply identical transformations
        image = self.resize(image)
        image = self.source_convas(image)
        image = self.cmnp(image, crop_pos_x=self.pos_x_rng(), crop_pos_y=self.pos_y_rng(), mirror=self.mirror_rng())
        # image = self.cmnp(image, mirror=self.mirror_rng())
        # image, label = self.crop([image, label])
        # image, label = self.mirror([image, label], vertical=self.mirror_rng())

        return image, self.target, self.camid

    def iter_setup(self, ):
        try:
            # print(self.dataset.i, self.dataset.n)
            images, labels, camids = self.iterator.next()
        except StopIteration:
            self.iterator = iter(self.dataset)
            images, labels, camids = self.dataset.next()
        self.feed_input(self.source, images, layout='HWC')
        self.feed_input(self.target, labels)
        self.feed_input(self.camid, camids)

    @property
    def size(self, ):
        return self.dataset.size


def get_loader(flist, batch_size=512, device_id=0):
    pipe = ReidPipeline(flist, batch_size=batch_size, num_threads=8, device_id=device_id)
    pipe.build()
    return DALIGenericIterator(pipe, ['images', 'labels', 'camids'], size=pipe.size, auto_reset=True)


def main():
    dataset = init_dataset('market1501')
    flist = dataset.train
    # test_list = FileListIterator(flist, 512)
    # for e in range(10):
    #     print(e)
    #     for d in test_list:
    #         continue
            # print(d)

    loader = get_loader(flist)
    for e in range(10):
        print(f'{e}')
        for i, data in enumerate(loader):
            for d in data:
                print(d['images'].shape)
                # print(d['labels'].shape)
            # print(d['rng'])


if __name__ == "__main__":
    main()
