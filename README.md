# ReID_baseline

A strong baseline (state-of-the-art) for person re-identification.

We support
- [x] easy dataset preparation
- [x] end-to-end training and evaluation
- [x] multi-GPU distributed training
- [x] fast data loader with prefetcher
- [ ] fast training speed with fp16
- [x] fast evaluation with cython
- [ ] support both image and video reid
- [x] multi-dataset training
- [x] cross-dataset evaluation
- [x] high modular management
- [x] state-of-the-art performance with simple model
- [x] high efficient backbone
- [x] advanced training techniques
- [x] various loss functions
- [x] tensorboard visualization 

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/L1aoXingyu/reid_baseline.git`
3. Install dependencies:
    - [pytorch 1.0.0+](https://pytorch.org/)
    - torchvision
    - tensorboard
    - [yacs](https://github.com/rbgirshick/yacs)
4. Prepare dataset

    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid_baseline
    mkdir datasets
    ```
    1. Download dataset to `datasets/` from [baidu pan](https://pan.baidu.com/s/1ntIi2Op) or [google driver](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
    2. Extract dataset. The dataset structure would like:
    ```bash
    datasets
        Market-1501-v15.09.15
            bounding_box_test/
            bounding_box_train/
    ```
5. Prepare pretrained model.
    If you use origin ResNet, you do not need to do anything. But if you want to use ResNet_ibn, you need to download pretrain model in [here](https://drive.google.com/open?id=1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S). And then you can put it in `~/.cache/torch/checkpoints` or anywhere you like.
    
    Then you should set this pretrain model path in `configs/softmax_triplet.yml`.

6. compile with cython to accelerate evalution
    ```bash
    cd csrc/eval_cylib; make
    ```

## Train
Most of the configuration files that we provide, you can run this command for training market1501
```bash
bash scripts/train_openset.sh
```

Or you can just run code below to modify your cfg parameters 
```bash
CUDA_VISIBLE_DEVICES='0,1' python tools/train.py -cfg='configs/softmax_triplet.yml' DATASETS.NAMES '("dukemtmc","market1501",)' SOLVER.IMS_PER_BATCH '256'
```

## Test
You can test your model's performance directly by running this command
```bash
CUDA_VISIBLE_DEVICES='0' python tools/test.py -cfg='configs/softmax_triplet.yml' DATASET.TEST_NAMES 'dukemtmc' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT '/save/trained_model/path'
```

## Experiment Results

| size=(256, 128) batch_size=64 (16 id x 4 imgs) |  |  |  |  |
| :------: | :-----: | :-----: | :--: | :---: |
|    softmax?   |    ✔︎   |   ✔︎   | ✔︎ | ✔︎ |
| label smooth? |  |  | ✔︎ | ✔︎ |
| triplet?   |        |  ✔︎    | ✔︎ | ✔︎ |
|    ibn?       |        |       |  ✔︎   |  ✔︎ |
|    gcnet?     |        |       |      |   ✔︎   |
|  Market1501   | 93.4 (82.9) | 94.2 (86.1) | 95.4 (87.9) | 95.2 (88.7) |
| DukeMTMC-reid | 84.7 (72.7) | 87.3 (76.0) | 89.5 (79.7) | 90.0 (80.2) |
|   CUHK03      | | | | |


🔥Any other tricks are welcomed！