# ReID_baseline
**The results in CUHK03 is wrong becuase I use the same evaluation protocol with market1501 and duke. I will fix this later. You can use the right evaluation protocol to get accurate result.**

Baseline model (with bottleneck) for person ReID (using softmax and triplet loss).

We support
- [x] easy dataset preparation
- [x] end-to-end training and evaluation
- [x] high modular management

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/L1aoXingyu/reid_baseline.git`
3. Install dependencies:
    - [pytorch 1.0](https://pytorch.org/)
    - torchvision
    - [ignite](https://github.com/pytorch/ignite)
    - [yacs](https://github.com/rbgirshick/yacs)
4. Prepare dataset

    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid_baseline
    mkdir data
    ```
    1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    2. Extract dataset and rename to `market1501`. The data structure would like:
    ```bash
    data
        market1501
            bounding_box_test/
            bounding_box_train/
    ```
5. Prepare pretrained model if you don't have
    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    Then it will automatically download model in `~/.torch/models/`, you should set this path in `config/defaults.py` for all training or set in every single training config file in `configs/`.

## Train
Most of the configuration files that we provide, you can run this command for training market1501
```bash
python3 tools/train.py --config_file='configs/softmax.yml' DATASETS.NAMES "('market1501')"
```

You can also modify your cfg parameters as follow
```bash
python3 tools/train.py --config_file='configs/softmax.yml' INPUT.SIZE_TRAIN '(256, 128)' INPUT.SIZE_TEST '(256, 128)'
```

## Test
You can test your model's performance directly by running this command
```bash
python3 tools/test.py --config_file='configs/softmax.yml' TEST.WEIGHT '/save/trained_model/path'
```

## Results

**network architecture**

<div align=center>
<img src='https://ws3.sinaimg.cn/large/006tNbRwly1fvh3ekjh12j315k0j4q58.jpg' width='500'>
</div>

| cfg | market1501 | dukemtmc |
| --- | -- | -- |
| softmax, size=(384, 128), batch_size=64 | 92.5 (79.4) |  84.6 (68.1) |
| softmax, size=(256, 128), batch_size=64 | 92.0 (80.4) |  84.1(68.4) |
| softmax_triplet, size=(384, 128), batch_size=128(32 id x 4 imgs) | 93.2 (82.5) |  86.4 (73.1) |
| softmax_triplet, size=(384, 128), batch_size=64(16 id x 4 imgs) | 93.8 (83.2) |  86.2 (72.9) |
| softmax_triplet, size=(256, 128), batch_size=64(16 id x 4 imgs) | 93.8 (85.3) | 86.0 (74.0) |
