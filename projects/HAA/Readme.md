# Towards Person ReID in Similar Clothings with Head-shoulder Information


## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yml>
```

## Evaluation

To evaluate the model in test set, run similarly:

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <configs.yaml> --eval-only MODEL.WEIGHTS model.pth
```

## Experimental Results

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: | 
| HSE | ImageNet | 95.8% | 89.5% | 

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | 
| :---: | :---: | :---: |:---: | 
| HSE | ImageNet | 89% | 80.4% | 

### Black_reid dataset

| Method | Pretrained | Rank@1 | mAP | 
| :---: | :---: | :---: |:---: | 
| HSE | ImageNet | 90.9%  | 83.8% | 

### White_reid dataset

| Method | Pretrained | Rank@1 | mAP | 
| :---: | :---: | :---: |:---: | 
| HSE | ImageNet | 95.3%  | 88.1% | 


