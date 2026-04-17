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

### Windows CUDA helper (RTX 3070 Ti)

Use the PowerShell helper to start a run from scratch or resume from the latest checkpoint:

`tools/run_temporal_windows_cuda.ps1`

Or use the batch wrapper (double-click-friendly), which forwards all args:

`tools/run_temporal_windows_cuda.bat`

Fresh run example:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_temporal_windows_cuda.ps1 `
  -Mode fresh `
  -InputWav "C:\data\input.wav" `
  -OutputWav "C:\data\output.wav" `
  -OutDir "C:\runs\nam_temporal_rtx3070ti"
```

Resume example (uses `OutDir\checkpoints\last.ckpt`):

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_temporal_windows_cuda.ps1 `
  -Mode resume `
  -InputWav "C:\data\input.wav" `
  -OutputWav "C:\data\output.wav" `
  -OutDir "C:\runs\nam_temporal_rtx3070ti"
```

Notes:

- The script sets `--device gpu` and defaults to `--precision 16-mixed`.
- In `fresh` mode, existing `OutDir` is archived with a timestamp.
- In `resume` mode, the script fails fast if `last.ckpt` is missing.

Or on Windows,
```
cd docs
make.bat html
```
