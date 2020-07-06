# Getting Started with Fastreid

## Prepare pretrained model

If you use origin ResNet, you do not need to do anything. But if you want to use ResNet-ibn or ResNeSt, you need to download pretrain model in [here](https://drive.google.com/open?id=1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S).
And then you need to put it in `~/.cache/torch/checkpoints` or anywhere you like.

Then you should set the pretrain model path in `configs/Base-bagtricks.yml`.

## Compile with cython to accelerate evalution

```bash
cd fastreid/evaluation/rank_cylib; make all
```

## Training & Evaluation in Command Line

We provide a script in "tools/train_net.py", that is made to train all the configs provided in fastreid.
You may want to use it as a reference to write your own training script.

To train a model with "train_net.py", first setup up the corresponding datasets following [datasets/README.md](https://github.com/JDAI-CV/fast-reid/tree/master/datasets), then run:

```bash
./tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml MODEL.DEVICE "cuda:0"
```

The configs are made for 1-GPU training.

If you want to train model with 4 GPUs, you can run:

```bash
./tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --num-gpus 4
```

To evaluate a model's performance, use

```bash
./tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --eval-only \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```

For more options, see `./tools/train_net.py -h`.
