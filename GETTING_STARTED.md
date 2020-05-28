# Getting Started with Fastreid

## Prepare pretrained model

If you use origin ResNet, you do not need to do anything. But if you want to use ResNet-ibn or ResNeSt, you need to download pretrain model in [here](https://drive.google.com/open?id=1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S).
And then you need to put it in `~/.cache/torch/checkpoints` or anywhere you like.

Then you should set the pretrain model path in `configs/Base-bagtricks.yml`.

## Compile with cython to accelerate evalution

```bash
cd fastreid/evaluation/rank_cylib; make all
```
