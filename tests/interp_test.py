import torch
from fastai.vision import *
from fastai.basic_data import *
from fastai.layers import *

import sys
sys.path.append('.')
from engine.interpreter import ReidInterpretation

from data import get_data_bunch
from modeling import build_model
from config import cfg
cfg.DATASETS.NAMES = ('market1501',)
cfg.DATASETS.TEST_NAMES = 'market1501'
cfg.MODEL.BACKBONE = 'resnet50'

data_bunch, test_labels, num_query = get_data_bunch(cfg)

model = build_model(cfg, 10)
model.load_params_wo_fc(torch.load('logs/2019.8.14/market/baseline/models/model_149.pth')['model'])
learn = Learner(data_bunch, model)

feats, _ = learn.get_preds(DatasetType.Test, activ=Lambda(lambda x: x))