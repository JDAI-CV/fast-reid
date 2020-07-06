# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn.functional as F
from collections import defaultdict
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.backends import cudnn
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.config import get_cfg

cudnn.benchmark = True


class Reid(object):

    def __init__(self, config_file):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = 'projects/bjzProject/logs/bjz/arcface_adam/model_final.pth'
        model = build_model(cfg)
        Checkpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)

        model.cuda()
        model.eval()
        self.model = model
        # self.model = torch.jit.load("reid_model.pt")
        # self.model.eval()
        # self.model.cuda()

        example = torch.rand(1, 3, 256, 128)
        example = example.cuda()
        traced_script_module = torch.jit.trace_module(model, {'inference': example})
        traced_script_module.save("reid_feat_extractor.pt")

    @classmethod
    def preprocess(cls, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = img[np.newaxis, :, :, :]
        data = torch.from_numpy(img).cuda().float()
        return data

    @torch.no_grad()
    def demo(self, img_path):
        data = self.preprocess(img_path)
        output = self.model.inference(data)
        feat = output.cpu().data.numpy()
        return feat

    # @torch.no_grad()
    # def extract_feat(self, dataloader):
    #     prefetcher = test_data_prefetcher(dataloader)
    #     feats = []
    #     labels = []
    #     batch = prefetcher.next()
    #     num_count = 0
    #     while batch[0] is not None:
    #         img, pid, camid = batch
    #         feat = self.model(img)
    #         feats.append(feat.cpu())
    #         labels.extend(np.asarray(pid))
    #
    #         # if num_count > 2:
    #             # break
    #         batch = prefetcher.next()
    #         # num_count += 1
    #
    #     feats = torch.cat(feats, dim=0)
    #     id_feats = defaultdict(list)
    #     for f, i in zip(feats, labels):
    #         id_feats[i].append(f)
    #     all_feats = []
    #     label_names = []
    #     for i in id_feats:
    #         all_feats.append(torch.stack(id_feats[i], dim=0).mean(dim=0))
    #         label_names.append(i)
    #
    #     label_names = np.asarray(label_names)
    #     all_feats = torch.stack(all_feats, dim=0)  # (n, 2048)
    #     all_feats = F.normalize(all_feats, p=2, dim=1)
    #     np.save('feats.npy', all_feats.cpu())
    #     np.save('labels.npy', label_names)
    #     cos = torch.mm(all_feats, all_feats.t()).numpy()  # (n, n)
    #     cos -= np.eye(all_feats.shape[0])
    #     f = open('check_cross_folder_similarity.txt', 'w')
    #     for i in range(len(label_names)):
    #         sim_indx = np.argwhere(cos[i] > 0.5)[:, 0]
    #         sim_name = label_names[sim_indx]
    #         write_str = label_names[i] + ' '
    #         # f.write(label_names[i]+'\t')
    #         for n in sim_name:
    #             write_str += (n + ' ')
    #             # f.write(n+'\t')
    #         f.write(write_str+'\n')
    #
    #
    # def prepare_gt(self, json_file):
    #     feat = []
    #     label = []
    #     with open(json_file, 'r') as f:
    #         total = json.load(f)
    #         for index in total:
    #             label.append(index)
    #             feat.append(np.array(total[index]))
    #     time_label = [int(i[0:10]) for i in label]
    #
    #     return np.array(feat), np.array(label), np.array(time_label)

    def compute_topk(self, k, feat, feats, label):

        # num_gallery = feats.shape[0]
        # new_feat = np.tile(feat,[num_gallery,1])
        norm_feat = np.sqrt(np.sum(np.square(feat), axis=-1))
        norm_feats = np.sqrt(np.sum(np.square(feats), axis=-1))
        matrix = np.sum(np.multiply(feat, feats), axis=-1)
        dist = matrix / np.multiply(norm_feat, norm_feats)
        # print('feat:',feat.shape)
        # print('feats:',feats.shape)
        # print('label:',label.shape)
        # print('dist:',dist.shape)

        index = np.argsort(-dist)

        # print('index:',index.shape)
        result = []
        for i in range(min(feats.shape[0], k)):
            print(dist[index[i]])
            result.append(label[index[i]])
        return result


if __name__ == '__main__':
    reid_sys = Reid(config_file='../../projects/bjzProject/configs/bjz.yml')
    img_path = '/export/home/lxy/beijingStationReID/reid_model/demo_imgs/003740_c5s2_1561733125170.000000.jpg'
    feat = reid_sys.demo(img_path)
    feat_extractor = torch.jit.load('reid_feat_extractor.pt')
    data = reid_sys.preprocess(img_path)
    feat2 = feat_extractor.inference(data)
    from ipdb import set_trace; set_trace()
    # imgs = os.listdir(img_path)
    # feats = {}
    # for i in range(len(imgs)):
    # feat = reid.demo(os.path.join(img_path, imgs[i]))
    # feats[imgs[i]] = feat
    # feat = reid.demo(os.path.join(img_path, 'crop_img0.jpg'))
    # out1 = feats['dog.jpg']
    # out2 = feats['kobe2.jpg']
    # innerProduct = np.dot(out1, out2.T)
    # cosineSimilarity = innerProduct / (np.linalg.norm(out1, ord=2) * np.linalg.norm(out2, ord=2))
    # print(f'cosine similarity is {cosineSimilarity[0][0]:.4f}')
