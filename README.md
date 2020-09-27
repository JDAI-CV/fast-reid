# FastReID

FastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a groud-up rewrite of the previous version, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## What's New

- [Sep 2020] Added the [person attribute recognition](https://github.com/JDAI-CV/fast-reid/tree/master/projects/attribute_recognition) based on fastreid. See `projects/attribute_recognition`.
- [Sep 2020] Automatic Mixed Precision training is supported with pytorch1.6 built-in `torch.cuda.amp`. Set `cfg.SOLVER.AMP_ENABLED=True` to switch it on.
- [Aug 2020] [Model Distillation](https://github.com/JDAI-CV/fast-reid/tree/master/projects/DistillReID) is supported, thanks for [guan'an wang](https://github.com/wangguanan)'s contribution.
- [Aug 2020] ONNX/TensorRT converter is supported.
- [Jul 2020] Distributed training with multiple GPUs, it trains much faster.
- [Jul 2020] `MAX_ITER` in config means `epoch`, it will auto scale to maximum iterations.
- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.
- Can be used as a library to support [different projects](https://github.com/JDAI-CV/fast-reid/tree/master/projects) on top of it. We'll open source more research projects in this way.
- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).

We write a [chinese blog](https://l1aoxingyu.github.io/blogpages/reid/2020/05/29/fastreid.html) about this toolbox.

## Installation

See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/docs/INSTALL.md).

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](https://github.com/JDAI-CV/fast-reid/blob/master/docs/GETTING_STARTED.md).

Learn more at out [documentation](). And see [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects) for some projects that are build on top of fastreid.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/docs/MODEL_ZOO.md).

## Deployment

We provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](https://github.com/JDAI-CV/fast-reid/blob/master/tools/deploy).

## License

Fastreid is released under the [Apache 2.0 license](https://github.com/JDAI-CV/fast-reid/blob/master/LICENSE).

## Citing Fastreid

If you use Fastreid in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
