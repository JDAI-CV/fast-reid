#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python3 main_reid.py train --save_dir='/DATA/pytorch-ckpt/market1501_softmax_triplet' \
--max_epoch=400 --eval_step=50 --model_name='softmax_triplet'
