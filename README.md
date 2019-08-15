# ReID_baseline

A strong baseline (state-of-the-art) for person re-identification.

We support
- [x] easy dataset preparation
- [x] end-to-end training and evaluation
- [ ] multi-GPU distributed training
- [ ] fast training speed with fp16
- [ ] support both image and video reid
- [x] multi-dataset training
- [x] cross-dataset evaluation
- [x] high modular management
- [x] state-of-the-art performance with simple model
- [ ] high efficient backbone
- [ ] advanced training techniques
- [ ] various loss functions
- [ ] visualization tools

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/L1aoXingyu/reid_baseline.git`
3. Install dependencies:
    - [pytorch 1.0.0+](https://pytorch.org/)
    - torchvision
    - [fastai](https://github.com/fastai/fastai)
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
5. Prepare pretrained model if you don't have
    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    Then it will automatically download model in `~/.cache/torch/checkpoints/`, you should set this path in `config/defaults.py` for all training or set in every single training config file in `configs/`.

## Train
Most of the configuration files that we provide, you can run this command for training market1501
```bash
bash scripts/train_market.sh
```

Or you can just run code below to modify your cfg parameters 
```bash
python3 tools/train.py -cfg='configs/softmax.yml' INPUT.SIZE_TRAIN '(256, 128)' INPUT.SIZE_TEST '(256, 128)'
```

## Test
You can test your model's performance directly by running this command
```bash
python3 tools/test.py --config_file='configs/softmax.yml' TEST.WEIGHT '/save/trained_model/path'
```

## Results


| cfg | market1501 | dukemtmc |
| --- | -- | -- |
| softmax_triplet, size=(256, 128), batch_size=64(16 id x 4 imgs) | 93.9 (85.9) | training |
