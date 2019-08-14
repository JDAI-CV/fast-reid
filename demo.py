# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.backends import cudnn

from modeling import Baseline

cudnn.benchmark = True

class Reid(object):

    def __init__(self):

        # self.cfg = self.prepare('config/softmax_triplet.yml')
        # self.num_classes = 413
        # self.model = Baseline('resnet50_ibn', 100, 1)
        # state_dict = torch.load('/export/home/lxy/reid_baseline/logs/2019.8.12/bj/ibn_lighting/models/model_119.pth')
        # self.model.load_params_wo_fc(state_dict['model'])
        # self.model.cuda()
        # self.model.eval()
        self.model = torch.jit.load("reid_model.pt")
        # self.model.eval()
        # self.model.cuda()
        
        # example = torch.rand(1, 3, 256, 128)
        # example = example.cuda()
        # traced_script_module = torch.jit.trace(self.model, example)
        # traced_script_module.save("reid_model.pt")

    def demo(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229,0.224,0.225]
        img = img.transpose((2,0,1)).astype(np.float32)
        img = img[np.newaxis,:,:,:]
        data = torch.from_numpy(img).cuda().float()
        output = self.model(data)
        feat = output.cpu().data.numpy()

        return feat
    
    def prepare_gt(self,json_file):
        feat = []
        label = []
        with open(json_file,'r') as f:
            total = json.load(f)
            for index in total:
                label.append(index)
                feat.append(np.array(total[index]))
        time_label = [int(i[0:10]) for i in label] 

        return np.array(feat),np.array(label),np.array(time_label)

    def compute_topk(self,k,feat,feats,label):
        
        #num_gallery = feats.shape[0]
        #new_feat = np.tile(feat,[num_gallery,1])
        norm_feat = np.sqrt(np.sum(np.square(feat),axis = -1))
        norm_feats = np.sqrt(np.sum(np.square(feats),axis = -1))
        matrix = np.sum(np.multiply(feat,feats),axis=-1)
        dist = matrix / np.multiply(norm_feat,norm_feats)
        #print('feat:',feat.shape)
        #print('feats:',feats.shape)
        #print('label:',label.shape)
        #print('dist:',dist.shape)
        
        index = np.argsort(-dist)

        #print('index:',index.shape)
        result = []
        for i in range(min(feats.shape[0],k)):
            print(dist[index[i]])
            result.append(label[index[i]])
        return result


if __name__ == '__main__':
    img_path = '/export/home/lxy/reid_demo/imgs'
    reid = Reid()
    img1 = ['1-1.png', '1-2.png', '1-3.png', '1-4.png', '1-5.png']
    img2 = ['2-1.png', '2-2.png']
    for i in range(len(img1)):
        for j in range(len(img2)):
            out1 = reid.demo(os.path.join(img_path, img1[i]))
            out2 = reid.demo(os.path.join(img_path, img2[j]))
            innerProduct = np.dot(out1, out2.T)
            cosineSimilarity = innerProduct / (np.linalg.norm(out1, ord=2) * np.linalg.norm(out2, ord=2))
            print('img {} and img {} cosine similarity is {:.4f}'.format(img1[i], img2[j], cosineSimilarity[0][0]))
