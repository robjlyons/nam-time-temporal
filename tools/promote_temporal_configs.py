#!/usr/bin/env python3
"""
Promote top sweep configs to long training runs.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import shutil
import subprocess


def _run(cmd: list[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_top_configs(leaderboard_csv: Path, top_k: int) -> list[str]:
    rows = list(csv.DictReader(leaderboard_csv.read_text().splitlines()))
    rows.sort(key=lambda r: float(r["nam_esr"]))
    return [r["name"] for r in rows[:top_k]]


def _resume_checkpoint(short_outdir: Path) -> Path:
    ckpt = short_outdir / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    return ckpt


def _read_cfg(short_outdir: Path) -> dict:
    cfg_path = short_outdir / "training_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return json.loads(cfg_path.read_text())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--promote-root", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--preview-every", type=int, default=1500)
    parser.add_argument("--eval-chunk-size", type=int, default=16384)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sweep_root = Path(args.sweep_root).resolve()
    promote_root = Path(args.promote_root).resolve()
    promote_root.mkdir(parents=True, exist_ok=True)

    selected = _load_top_configs(Path(args.leaderboard).resolve(), int(args.top_k))
    summary: list[dict] = []
    for name in selected:
        short_outdir = sweep_root / name
        cfg = _read_cfg(short_outdir)
        outdir = promote_root / f"{name}_long"
        outdir.mkdir(parents=True, exist_ok=True)
        resume_ckpt = _resume_checkpoint(short_outdir)
        cfg_copy_path = outdir / "source_short_config.json"
        cfg_copy_path.write_text(json.dumps(cfg, indent=2))
        shutil.copy2(resume_ckpt, outdir / "resume_from.ckpt")

        train_cmd = [
            "python",
            "train.py",
            "--input",
            args.input,
            "--output",
            args.target,
            "--outdir",
            str(outdir),
            "--resume",
            str(resume_ckpt),
            "--steps",
            str(args.max_steps),
            "--batch-size",
            str(cfg["batch_size"]),
            "--context",
            str(cfg["context_samples"]),
            "--target",
            str(cfg["target_samples"]),
            "--hidden-size",
            str(cfg["hidden_size"]),
            "--train-burn-in",
            str(cfg.get("train_burn_in") or 0),
            "--train-truncate",
            str(cfg.get("train_truncate") or 0),
            "--local-layers",
            str(cfg["local_layers"]),
            "--learning-rate",
            str(cfg["learning_rate"]),
            "--esr-denominator-floor",
            str(cfg.get("esr_denominator_floor") or 0.0),
            "--mrstft-weight",
            str(cfg["mrstft_weight"] if cfg["mrstft_weight"] is not None else 0.0),
            "--epoch-steps",
            str(cfg["epoch_steps"]),
            "--val-check-interval",
            str(cfg["val_check_interval"]),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--preview-every",
            str(args.preview_every),
            "--num-workers",
            str(cfg["num_workers"]),
            "--precision",
            str(cfg["precision"]),
            "--device",
            str(cfg["device"]),
        ]
        if not cfg.get("deterministic_validation", True):
            train_cmd.append("--no-deterministic-validation")
        if cfg.get("lr_scheduler"):
            train_cmd.extend(
                [
                    "--lr-scheduler",
                    str(cfg["lr_scheduler"]),
                    "--lr-factor",
                    str(cfg["lr_factor"]),
                    "--lr-patience",
                    str(cfg["lr_patience"]),
                    "--lr-min",
                    str(cfg["lr_min"]),
                ]
            )
        if cfg.get("force_mono", False):
            train_cmd.append("--force-mono")
        if not cfg.get("enable_logger", True):
            train_cmd.append("--no-logger")

        _run(train_cmd, cwd=repo_root)
        eval_cmd = [
            "python",
            "evaluate.py",
            "--checkpoint",
            str(outdir / "model.nam"),
            "--input",
            args.input,
            "--target",
            args.target,
            "--chunk-size",
            str(args.eval_chunk_size),
            "--outdir",
            str(outdir / "eval_model_nam"),
        ]
        _run(eval_cmd, cwd=repo_root)
        metrics = json.loads((outdir / "eval_model_nam" / "metrics.json").read_text())
        summary.append({"name": name, "outdir": str(outdir), "nam_esr": metrics["esr"]})

    (promote_root / "promotion_summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] wrote", promote_root / "promotion_summary.json")


if __name__ == "__main__":
    main()
