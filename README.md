<h1 align="center">WHIP</h1>
<h3 align="center">Towards Real-World Wearable Motion Reconstruction</h3>
<h5 align="center">ECCV 2026</h5>

This repository contains the official implementation of the paper "WHIP: Towards Real-World Wearable Motion Reconstruction".

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Training](#training)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Pre-requisites

- [uv](https://docs.astral.sh/uv/) for dependency management
- Python 3.13 or newer
- A CUDA-capable GPU is recommended for training and evaluation

## Installation

From the repository root, install the locked dependencies with:

```bash
uv sync
```

## Dataset

Download the WHIP dataset from [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.ZGVC7M) and extract the zip file locally.

## Evaluation

Evaluate a checkpoint with:

```bash
uv run -m whip.eval /path/to/whip-dataset/data /path/to/model.pt --combo all
```

Useful options:

- `--split`: one of `test`, `unseen_actor`, `unseen_actions`, `unseen_both`, or `unseen_actor_or_actions`
- `--combo`: comma-separated sensors to condition on, or `all`
- `--samples`: number of stochastic samples to average

For example, `--combo insoles,watch_left,label` evaluates with insoles, the left watch, and action labels.

## Training

Train the default WHIP model with:

```bash
uv run -m whip.train /path/to/whip-dataset/data --out logs/whip --name whip
```

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@InProceedings{Camiletto_2026_WHIP,
  author    = {Boscolo Camiletto, Andrea and Dabral, Rishabh and Alvarado, Eduardo and Beeler, Thabo and Habermann, Marc and Theobalt, Christian},
  title     = {Towards Real-World Wearable Motion Reconstruction},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

This implementation builds on open-source software including [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), [Typer](https://typer.tiangolo.com/), and [uv](https://docs.astral.sh/uv/).

## License

This repository is released under the Creative Commons Attribution-NonCommercial 4.0 International license. See [LICENSE](LICENSE) for details.
