#!/usr/bin/env python
"""
Temporal-effects trainer entrypoint.
"""

from argparse import ArgumentParser
from pathlib import Path

from nam.training.config import TemporalTrainingConfig
from nam.training.engine import train_temporal_model


def _parse_args():
    p = ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--target", type=int, default=8192)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--preview-every", type=int, default=1000)
    p.add_argument(
        "--val-check-interval",
        type=int,
        default=250,
        help="Validation every N train batches; must be <= ceil(epoch_steps/batch_size).",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-persistent-workers", action="store_true")
    p.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed"],
    )
    p.add_argument(
        "--mrstft-weight",
        type=float,
        default=2e-4,
        help="Set to 0 to disable MRSTFT for faster exploratory runs.",
    )
    p.add_argument("--hidden-size", type=int, default=48, help="TemporalHybrid LSTM hidden size.")
    p.add_argument("--local-layers", type=int, default=2, help="Depth of local conv stack.")
    p.add_argument(
        "--epoch-steps",
        type=int,
        default=2000,
        help="Train DataLoader length per epoch (random windows). Affects val length too.",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=1024,
        help="Train-window start jitter (+/- overlap) in samples.",
    )
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--force-mono", action="store_true")
    p.add_argument(
        "--m1-tuned",
        action="store_true",
        help="Apply conservative Apple Silicon speed defaults.",
    )
    return p.parse_args()


def main():
    a = _parse_args()
    if a.m1_tuned:
        # Conservative defaults for 16GB unified-memory Macs.
        if a.num_workers == 2:
            a.num_workers = 2
        if a.prefetch_factor == 2:
            a.prefetch_factor = 2
        if a.preview_every == 1000:
            a.preview_every = 1500
        if a.checkpoint_every == 500:
            a.checkpoint_every = 1000
        if a.precision == "32-true":
            # Keep numerically stable default unless user opts in.
            a.precision = "32-true"
    cfg = TemporalTrainingConfig(
        input_wav=Path(a.input),
        output_wav=Path(a.output),
        outdir=Path(a.outdir),
        max_steps=a.steps,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        persistent_workers=not a.no_persistent_workers,
        prefetch_factor=a.prefetch_factor,
        context_samples=a.context,
        target_samples=a.target,
        overlap_samples=a.overlap,
        epoch_steps=a.epoch_steps,
        hidden_size=a.hidden_size,
        local_layers=a.local_layers,
        val_check_interval=a.val_check_interval,
        checkpoint_every_n_steps=a.checkpoint_every,
        preview_every_n_steps=a.preview_every,
        log_every_n_steps=a.log_every,
        precision=a.precision,
        mrstft_weight=a.mrstft_weight if a.mrstft_weight > 0 else None,
        resume=Path(a.resume) if a.resume else None,
        device=a.device,
        force_mono=a.force_mono,
    )
    train_temporal_model(cfg)


if __name__ == "__main__":
    main()
