# FastReID

FastReID is a research platform that implements state-of-the-art re-identification algorithms. 

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/L1aoXingyu/fast-reid.git`
3. Install dependencies:
    - [pytorch 1.0.0+](https://pytorch.org/)
    - torchvision
    - tensorboard
    - [yacs](https://github.com/rbgirshick/yacs)
4. Prepare dataset
    Create a directory to store reid datasets under projects, for example

    ```bash
    cd fast-reid/projects/StrongBaseline
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

    Then you should set the pretrain model path in `configs/baseline_market1501.yml`.

6. compile with cython to accelerate evalution

    ```bash
    cd fastreid/evaluation/rank_cylib; make all
    ```

## Model Zoo and Baselines

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 93.6% | 85.1% | 58.1% |
| BagTricks + Ibn-a | ImageNet | 94.8% | 87.3% | 63.5% |
| AGW |  ImageNet | 94.9% | 87.4% | 63.1% |


### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 86.1% | 75.9% | 38.7% |
| BagTricks + Ibn-a | ImageNet | 89.0% | 78.8% | 43.6% |
| AGW |  ImageNet | 88.9% | 79.1% | 43.2% |


### MSMT17 dataset

| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BagTricks | ImageNet | 70.4%  | 47.5% | 9.6% |
| BagTricks + Ibn-a | ImageNet | 76.9% | 55.0% | 13.5% |
| AGW | ImageNet | 75.6% | 52.6% | 11.9% |
