# coding: utf-8
import os
from collections import defaultdict
import sys
import shutil

sys.path.append('')

from fastreid.utils.env import seed_all_rng
from fastreid.data.datasets import DATASET_REGISTRY

import projects.Shoe.shoe.data

seed_all_rng(0)

save_root = 'debug/neg_aug'
if os.path.exists(save_root):
    shutil.rmtree(save_root)
os.mkdir(save_root)

root = '/data97/bijia/shoe/'
img_root=os.path.join(root, 'shoe_crop_all_images')
anno_path=os.path.join(root, 'labels/1102/train_1102.json')
dataset = DATASET_REGISTRY.get('PairDataset')(img_root=img_root, anno_path=anno_path, transform=None, mode='train')

pos_imgs = []
neg_imgs = []
for i in range(100):
	data = dataset[100]
	img1 = data['img1']
	img2 = data['img2']
	target = data['target']
	
	if target == 0:
		pos_imgs.append(img1)
		neg_imgs.append(img2)
	else:
		pos_imgs.append(img1)
		pos_imgs.append(img2)

pos_dict = defaultdict(list)
for img in pos_imgs:
	pos_dict[img.size].append(img)

for i, k in enumerate(pos_dict.keys()):
    img = pos_dict[k][0]
    img.save(os.path.join(save_root, 'p-' + str(i) + '.jpg'))

for i, img in enumerate(neg_imgs):
    img.save(os.path.join(save_root, 'n-' + str(i) + '.jpg'))
