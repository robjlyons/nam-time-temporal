#!/usr/bin/env python3
"""
Compute quick ESR feasibility baselines for temporal training targets.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate import _load_model
from nam.data_streaming import load_audio_pair
from nam.inference.offline import process_chunked
from nam.models.losses import esr


def _active_segment_start(y: torch.Tensor, window: int, quantile: float) -> int:
    # y: (C, N)
    wet = y.mean(dim=0)
    n = wet.numel()
    if n <= window:
        return 0
    step = max(1, window // 4)
    windows = wet.unfold(0, window, step)
    energy = torch.mean(windows**2, dim=1)
    threshold = torch.quantile(energy, torch.tensor(float(quantile)))
    active_idx = torch.nonzero(energy >= threshold).flatten()
    if active_idx.numel() == 0:
        best = int(torch.argmax(energy))
    else:
        best = int(active_idx[0])
    return int(best * step)


def _run_model(model_path: Path, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    model = _load_model(model_path)
    y_hat_channels = []
    for c in range(x.shape[0]):
        yc = process_chunked(model, x[c], chunk_size=chunk_size).detach().cpu()
        y_hat_channels.append(yc)
    return torch.stack(y_hat_channels, dim=0)


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--model", default=None, help=".nam or .ckpt path")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--chunk-size", type=int, default=16384)
    p.add_argument("--active-window", type=int, default=4096)
    p.add_argument("--active-quantile", type=float, default=0.9)
    args = p.parse_args()

    pair = load_audio_pair(args.input, args.target, target_sample_rate=None, force_mono=False)
    x = torch.from_numpy(pair.dry).float()
    y = torch.from_numpy(pair.wet).float()

    n = min(x.shape[-1], y.shape[-1])
    x = x[:, :n]
    y = y[:, :n]

    zero = torch.zeros_like(y)
    full_zero_esr = float(esr(zero, y))

    start = _active_segment_start(y, int(args.active_window), float(args.active_quantile))
    stop = min(n, start + int(args.active_window))
    active_zero_esr = float(esr(zero[:, start:stop], y[:, start:stop]))

    out = {
        "sample_rate": int(pair.sample_rate),
        "num_samples": int(n),
        "active_window": int(args.active_window),
        "active_start": int(start),
        "active_stop": int(stop),
        "full_zero_esr": full_zero_esr,
        "active_zero_esr": active_zero_esr,
    }

    if args.model:
        y_hat = _run_model(Path(args.model), x, int(args.chunk_size))
        m = min(y_hat.shape[-1], y.shape[-1])
        y_hat = y_hat[:, :m]
        y_ref = y[:, :m]
        out["full_model_esr"] = float(esr(y_hat, y_ref))
        out["active_model_esr"] = float(esr(y_hat[:, start:stop], y_ref[:, start:stop]))
        out["model_path"] = str(Path(args.model).resolve())

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
