# NAM: Neural Amp Modeler

[![Build](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml)

This repository handles training models and exporting them to .nam files.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For more information about the NAM ecosystem please check out https://www.neuralampmodeler.com/.

## Documentation
Online documentation can be found here: 
https://neural-amp-modeler.readthedocs.io

To build the documentation locally on a Linux system:
```bash
cd docs
make html
```

## Temporal effects workflow

This fork includes an extended temporal pipeline for delay/reverb captures:

- `train.py` for long-context temporal training with checkpoints/resume and audio previews
- `notebooks/temporal_colab.ipynb` — Colab/Kaggle-oriented notebook (GPU) mirroring the official three-step layout for this temporal trainer
- `evaluate.py` for ESR/STFT scoring and rendered output audio
- `webapp/backend` and `webapp/frontend` for browser-based training UX

Or on Windows,
```
cd docs
make.bat html
```
