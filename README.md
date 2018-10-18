# ReID_baseline
Baseline model (with bottleneck) for person ReID (using softmax and triplet loss). This is PyTorch version, [mxnet version](https://github.com/L1aoXingyu/reid_baseline_gluon) has a better result and more SOTA methods.

We support
- multi-GPU training
- easy dataset preparation
- end-to-end training and evaluation

## Get Started
1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/L1aoXingyu/reid_baseline.git`
3. Install dependencies:
    - [pytorch 0.4](https://pytorch.org/)
    - torchvision
    - tensorflow (for tensorboard)
    - [tensorboardX](https://github.com/lanpa/tensorboardX)
4. Prepare dataset
    
    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid_baseline
    mkdir data
    ```
    1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    2. Extract dataset and rename to `market1501`. The data structure would like:
    ```
    market1501/
        bounding_box_test/
        bounding_box_train/
    ```
5. Prepare pretrained model if you don't have
    ```python
    from torchvision import models
    models.resnet50(pretrained=True)
    ```
    Then it will automatically download model in `~.torch/models/`, you should set this path in `config.py`

## Train
You can run 
```bash
bash scripts/train_triplet_softmax.sh
```
in `reid_baseline` folder if you want to train with softmax and triplet loss. You can find others train scripts in `scripts`.

## Results

**network architecture**

<img src='https://ws3.sinaimg.cn/large/006tNbRwly1fvh3ekjh12j315k0j4q58.jpg' width='500'>

| config | Market1501 |
| --- | -- | 
| bs(32) size(384,128) softmax | 92.2 (78.5) |  
| bs(64) size(384,128) softmax | 92.5 (79.6) | 
| bs(32) size(256,128) softmax | 92.0 (78.4) | 
| bs(64) size(256,128) softmax | 91.7 (78.3) | 
| bs(128) size(256,128) softmax | 91.2 (77.4) | 
| triplet(p=32,k=4) size(256,128) | 88.3 (73.8) | 
| triplet(p=16,k=4)+softmax size(384,128) | 93.1 (82.0) | 
| triplet(p=24,k=4)+softmax size(384,128) | 91.7 (79.0) | 

