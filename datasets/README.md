# Setup Buildin Dataset

Fastreid has buildin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable `FASTREID_DATASETS`. Under this directory, fastreid expects to find datasets in the structure described below.

You can set the location for builtin datasets by `export FASTREID_DATASETS=/path/to/datasets/`. If left unset, the default is `datasets/` relative to your current working directory.

The [model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md) contains configs and models that use these buildin datasets.

## Expected dataset structure for [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)

1. Download dataset to `datasets/` from [baidu pan](https://pan.baidu.com/s/1ntIi2Op) or [google driver](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
2. Extract dataset. The dataset structure would like:

```bash
datasets/
    Market-1501-v15.09.15/
        bounding_box_test/
        bounding_box_train/
```

## Expected dataset structure for [DukeMTMC-reID](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.pdf)

1. Download datasets to `datasets/`
2. Extract dataset. The dataset structure would like:

```bash
datasets/
    DukeMTMC-reID/
        bounding_box_train/
        bounding_box_test/
```

## Expected dataset structure for [MSMT17](https://arxiv.org/abs/1711.08565)

1. Download datasets to `datasets/`
2. Extract dataset. The dataset structure would like:

```bash
datasets/
    MSMT17_V2/
        mask_train_v2/
        mask_test_v2/
```
