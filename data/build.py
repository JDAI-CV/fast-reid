# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import fast_collate_fn, test_collate_fn
from .datasets import ImageDataset
from .datasets import init_dataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms, build_mask_transforms


# def _process_bj_dir(dir_path, recursive=False):
#     img_paths = []
#     if recursive:
#         id_dirs = os.listdir(dir_path)
#         for d in id_dirs:
#             img_paths.extend(glob.glob(os.path.join(dir_path, d, '*.jpg')))
#     else:
#         img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
#
#     pattern = re.compile(r'([-\d]+)_c(\d*)')
#     v_paths = []
#     for img_path in img_paths:
#         try:
#             pid, camid = map(str, pattern.search(img_path).groups())
#         except:
#             from ipdb import set_trace; set_trace()
#             # import shutil
#             # if ' ' in img_path:
#             #     root_path = '/'.join(img_path.split('/')[:-1])
#             #     img_name = img_path.split('/')[-1]
#             #     new_img_name = img_name.split(' ')
#             #     new_img_name = new_img_name[0]+new_img_name[1]
#             #     shutil.move(img_path, os.path.join(root_path, new_img_name))
#             # else:
#             #     from ipdb import set_trace; set_trace()
#             #     root_path = '/'.join(img_path.split('/')[:-1])
#             #     img_name = img_path.split('/')[-1]
#             #     new_img_name = img_name.split('w')
#             #     new_img_name = new_img_name[0]+new_img_name[1]
#             #     shutil.move(img_path, os.path.join(root_path, new_img_name))
#         # pid = int(pid)
#         # if pid == -1: continue  # junk images are just ignored
#         v_paths.append([img_path, pid, camid])
#     return v_paths
#
#
# def _process_bj_test_dir(dir_path, recursive=False):
#     img_paths = []
#     if recursive:
#         id_dirs = os.listdir(dir_path)
#         for d in id_dirs:
#             img_paths.extend(glob.glob(os.path.join(dir_path, d, '*.jpg')))
#     else:
#         img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
#
#     pattern = re.compile(r'([-\d]+)_c(\d*)')
#     v_paths = []
#     for img_path in img_paths:
#         pid, camid = map(int, pattern.search(img_path).groups())
#         # pid = int(pid)
#         # if pid == -1: continue  # junk images are just ignored
#         v_paths.append([img_path, pid, camid])
#     return v_paths


def get_dataloader(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    mask_tfms = build_mask_transforms(cfg)
    val_tfms = build_transforms(cfg, is_train=False)

    print('prepare training set ...')
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = init_dataset(d, return_mask=cfg.INPUT.USE_MASK)
        train_img_items.extend(dataset.train)
    # for d in ['market1501', 'dukemtmc', 'msmt17']:
    #     dataset = init_dataset(d, combineall=True)
    #     train_img_items.extend(dataset.train)
    # bj_data = init_dataset('bjstation')
    # train_img_items.extend(bj_data.train)
    print('prepare test set ...')
    dataset = init_dataset(cfg.DATASETS.TEST_NAMES, return_mask=cfg.INPUT.USE_MASK)
    query_names, gallery_names = dataset.query, dataset.gallery

    tng_set = ImageDataset(train_img_items, tng_tfms, mask_tfms, relabel=True, return_mask=cfg.INPUT.USE_MASK)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # num_workers = 0
    data_sampler = None
    if cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler(tng_set.img_items, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)

    tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=(data_sampler is None),
                                num_workers=num_workers, sampler=data_sampler,
                                collate_fn=fast_collate_fn, pin_memory=True, drop_last=True)

    val_set = ImageDataset(query_names + gallery_names, val_tfms, relabel=False, return_mask=False)
    val_dataloader = DataLoader(val_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
                                collate_fn=fast_collate_fn, pin_memory=True)
    return tng_dataloader, val_dataloader, tng_set.c, len(query_names)


def get_test_dataloader(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    val_tfms = build_transforms(cfg, is_train=False)

    print('prepare test set ...')
    dataset = init_dataset(cfg.DATASETS.TEST_NAMES)
    query_names, gallery_names = dataset.query, dataset.gallery

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # train_img_items = list()
    # for d in cfg.DATASETS.NAMES:
    #     dataset = init_dataset(d)
    #     train_img_items.extend(dataset.train)

    # tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)

    tng_set = ImageDataset(query_names+gallery_names, tng_tfms, False)
    tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
                                num_workers=num_workers, collate_fn=fast_collate_fn, pin_memory=True, drop_last=True)
    test_set = ImageDataset(query_names + gallery_names, val_tfms, relabel=False)
    test_dataloader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
                                 collate_fn=fast_collate_fn, pin_memory=True)

    return tng_dataloader, test_dataloader, len(query_names)


def get_check_dataloader():
    import torchvision.transforms as T
    val_tfms = T.Compose([T.Resize((256, 128))])
    dataset = init_dataset('bjstation')
    train_names = dataset.train
    check_set = ImageDataset(train_names, val_tfms, relabel=False)
    check_loader = DataLoader(check_set, 512, shuffle=False, num_workers=16, collate_fn=test_collate_fn, pin_memory=True)
    return check_loader

