#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch
import wavio

from nam.data_streaming import load_audio_pair
from nam.inference.offline import process_chunked
from nam.models import init_from_nam
from nam.models.temporal_hybrid import TemporalHybrid
from nam.models.losses import esr
from nam.models.losses import multi_resolution_stft_loss
from nam.train.lightning_module import LightningModule


def _write_wav(path: Path, x: np.ndarray, sample_rate: int):
    x = np.clip(x, -1.0, 1.0)
    wavio.write(str(path), (x * (2**15 - 1)).astype(np.int16), sample_rate)


def _load_model(checkpoint_path: Path):
    if checkpoint_path.suffix.lower() == ".nam":
        with checkpoint_path.open("r") as fp:
            d = json.load(fp)
        model = init_from_nam(d)
        model.eval()
        return model
    try:
        lm = LightningModule.load_from_checkpoint(str(checkpoint_path))
        lm.eval()
        return lm.net
    except TypeError:
        # Some checkpoints don't include Lightning hyperparameters.
        # Reconstruct a compatible net from state_dict as a fallback.
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt["state_dict"]
        net_sd = {k.removeprefix("_net."): v for k, v in state_dict.items() if k.startswith("_net.")}
        hidden_size = int(net_sd["_lstm._core.weight_hh_l0"].shape[1])
        local_channels = int(net_sd["_local.0.weight"].shape[0])
        local_kernel_size = int(net_sd["_local.0.weight"].shape[2])
        # Count conv layers with kernel > 1 in local stack.
        local_layers = sum(
            1
            for k, v in net_sd.items()
            if k.endswith(".weight") and k.startswith("_local.") and v.ndim == 3 and v.shape[2] > 1
        )
        model = TemporalHybrid(
            hidden_size=hidden_size,
            local_channels=local_channels,
            local_kernel_size=local_kernel_size,
            local_layers=max(1, local_layers),
            fuse_alpha=1.0,
        )
        model.load_state_dict(net_sd, strict=False)
        model.sample_rate = ckpt.get("sample_rate")
        model.eval()
        return model


def _parse_args():
    p = ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--outdir", default="eval_out")
    p.add_argument("--chunk-size", type=int, default=16384)
    return p.parse_args()


def main():
    a = _parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pair = load_audio_pair(a.input, a.target, target_sample_rate=None, force_mono=False)
    model = _load_model(Path(a.checkpoint))
    sr = pair.sample_rate
    x = torch.from_numpy(pair.dry).float()
    y = torch.from_numpy(pair.wet).float()

    y_hat_channels = []
    for c in range(x.shape[0]):
        yc = process_chunked(model, x[c], chunk_size=a.chunk_size).detach().cpu()
        y_hat_channels.append(yc)
    y_hat = torch.stack(y_hat_channels, dim=0)
    n = min(y.shape[-1], y_hat.shape[-1], x.shape[-1])
    x = x[:, :n]
    y = y[:, :n]
    y_hat = y_hat[:, :n]

    metrics = {
        "esr": float(esr(y_hat, y)),
        "stft": float(multi_resolution_stft_loss(y_hat, y)),
        "num_samples": int(n),
        "sample_rate": int(sr),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _write_wav(outdir / "input.wav", x[0].numpy(), sr)
    _write_wav(outdir / "target.wav", y[0].numpy(), sr)
    _write_wav(outdir / "model_output.wav", y_hat[0].numpy(), sr)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
