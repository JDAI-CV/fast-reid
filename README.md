# FastReID

FastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a groud-up rewrite of the previous verson, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## What's New

- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).
- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.
- Can be used as a library to support [different projects](https://github.com/JDAI-CV/fast-reid/tree/master/projects) on top of it. We'll open source more research projects in this way.
- It trains much faster.

See our [zhihu blog]() to learn more about fastreid.

## Installation

See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](https://github.com/JDAI-CV/fast-reid/blob/master/GETTING_STARTED.md).

Learn more at out [documentation](). And see [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects) for some projects that are build on top of fastreid.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md).

## License

Fastreid is released under the [Apache 2.0 license](https://github.com/JDAI-CV/fast-reid/blob/master/LICENSE).
