#!/usr/bin/env python3
"""
Run temporal training sweeps with a canonical offline evaluation protocol.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import re
import subprocess
from typing import Any


CKPT_NAME_RE = re.compile(r"step\d+-val(?P<val>\d+\.\d+)\.ckpt$")


def _run(cmd: list[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _best_checkpoint(checkpoint_dir: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for p in checkpoint_dir.glob("step*-val*.ckpt"):
        m = CKPT_NAME_RE.match(p.name)
        if m is None:
            continue
        candidates.append((float(m.group("val")), p))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def _evaluate(
    repo_root: Path,
    checkpoint: Path,
    input_wav: Path,
    target_wav: Path,
    outdir: Path,
    chunk_size: int,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "python",
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_wav),
            "--target",
            str(target_wav),
            "--outdir",
            str(outdir),
            "--chunk-size",
            str(chunk_size),
        ],
        cwd=repo_root,
    )
    return json.loads((outdir / "metrics.json").read_text())


def _val_check_interval(epoch_steps: int, batch_size: int, preferred: int) -> int:
    n_batches = max(1, (epoch_steps + batch_size - 1) // batch_size)
    return min(preferred, n_batches)


def _train_and_eval(
    repo_root: Path,
    input_wav: Path,
    target_wav: Path,
    run_root: Path,
    base: dict[str, Any],
    exp: dict[str, Any],
    chunk_size: int,
) -> dict[str, Any]:
    cfg = {**base, **exp}
    name = cfg["name"]
    outdir = run_root / name
    outdir.mkdir(parents=True, exist_ok=True)
    val_check = _val_check_interval(
        int(cfg["epoch_steps"]), int(cfg["batch_size"]), int(cfg["val_check_interval"])
    )
    train_cmd = [
        "python",
        "train.py",
        "--input",
        str(input_wav),
        "--output",
        str(target_wav),
        "--outdir",
        str(outdir),
        "--steps",
        str(cfg["steps"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--context",
        str(cfg["context"]),
        "--target",
        str(cfg["target"]),
        "--hidden-size",
        str(cfg["hidden_size"]),
        "--train-burn-in",
        str(cfg.get("train_burn_in", 0)),
        "--train-truncate",
        str(cfg.get("train_truncate", 0)),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--esr-denominator-floor",
        str(cfg.get("esr_denominator_floor", 0.0)),
        "--mrstft-weight",
        str(cfg["mrstft_weight"]),
        "--epoch-steps",
        str(cfg["epoch_steps"]),
        "--val-check-interval",
        str(val_check),
        "--checkpoint-every",
        str(cfg["checkpoint_every"]),
        "--preview-every",
        str(cfg["preview_every"]),
        "--num-workers",
        str(cfg["num_workers"]),
        "--precision",
        str(cfg["precision"]),
        "--device",
        str(cfg["device"]),
    ]
    if cfg.get("force_mono", False):
        train_cmd.append("--force-mono")
    if cfg.get("no_logger", True):
        train_cmd.append("--no-logger")
    if cfg.get("lr_scheduler", "none") != "none":
        train_cmd.extend(
            [
                "--lr-scheduler",
                str(cfg["lr_scheduler"]),
                "--lr-factor",
                str(cfg.get("lr_factor", 0.5)),
                "--lr-patience",
                str(cfg.get("lr_patience", 6)),
                "--lr-min",
                str(cfg.get("lr_min", 1e-6)),
            ]
        )

    _run(train_cmd, cwd=repo_root)

    best_ckpt = _best_checkpoint(outdir / "checkpoints")
    eval_best = _evaluate(
        repo_root=repo_root,
        checkpoint=best_ckpt,
        input_wav=input_wav,
        target_wav=target_wav,
        outdir=outdir / "eval_best_ckpt",
        chunk_size=chunk_size,
    )
    nam_path = outdir / "model.nam"
    eval_nam = _evaluate(
        repo_root=repo_root,
        checkpoint=nam_path,
        input_wav=input_wav,
        target_wav=target_wav,
        outdir=outdir / "eval_model_nam",
        chunk_size=chunk_size,
    )
    result = {
        "name": name,
        "outdir": str(outdir),
        "best_checkpoint": str(best_ckpt),
        "val_check_interval_used": val_check,
        "checkpoint_esr": float(eval_best["esr"]),
        "checkpoint_stft": float(eval_best["stft"]),
        "nam_esr": float(eval_nam["esr"]),
        "nam_stft": float(eval_nam["stft"]),
        "config": cfg,
    }
    (outdir / "experiment_result.json").write_text(json.dumps(result, indent=2))
    return result


def _default_experiments() -> list[dict[str, Any]]:
    return [
        {"name": "baseline"},
        {"name": "context16384", "context": 16384},
        {"name": "truncate4096", "train_truncate": 4096, "train_burn_in": 2048},
        {"name": "hidden96", "hidden_size": 96},
        {"name": "mrstft0", "mrstft_weight": 0.0},
        {"name": "lr1e4", "learning_rate": 1e-4},
    ]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--plan-json", type=str, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--target-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--train-burn-in", type=int, default=0)
    parser.add_argument("--train-truncate", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--esr-denominator-floor", type=float, default=1e-8)
    parser.add_argument("--mrstft-weight", type=float, default=2e-4)
    parser.add_argument("--epoch-steps", type=int, default=2000)
    parser.add_argument("--val-check-interval", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--preview-every", type=int, default=1500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--with-logger", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_wav = Path(args.input).resolve()
    target_wav = Path(args.target).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    base = {
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "context": int(args.context),
        "target": int(args.target_size),
        "hidden_size": int(args.hidden_size),
        "train_burn_in": int(args.train_burn_in),
        "train_truncate": int(args.train_truncate),
        "learning_rate": float(args.learning_rate),
        "esr_denominator_floor": float(args.esr_denominator_floor),
        "mrstft_weight": float(args.mrstft_weight),
        "epoch_steps": int(args.epoch_steps),
        "val_check_interval": int(args.val_check_interval),
        "checkpoint_every": int(args.checkpoint_every),
        "preview_every": int(args.preview_every),
        "num_workers": int(args.num_workers),
        "precision": str(args.precision),
        "device": str(args.device),
        "no_logger": not bool(args.with_logger),
    }
    experiments = (
        json.loads(Path(args.plan_json).read_text())
        if args.plan_json is not None
        else _default_experiments()
    )
    if args.only:
        wanted = set(args.only)
        experiments = [e for e in experiments if e["name"] in wanted]

    results: list[dict[str, Any]] = []
    for exp in experiments:
        results.append(
            _train_and_eval(
                repo_root=repo_root,
                input_wav=input_wav,
                target_wav=target_wav,
                run_root=run_root,
                base=base,
                exp=exp,
                chunk_size=args.chunk_size,
            )
        )

    results.sort(key=lambda r: r["nam_esr"])
    summary_path = run_root / "leaderboard.json"
    summary_path.write_text(json.dumps(results, indent=2))

    csv_path = run_root / "leaderboard.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "name",
                "nam_esr",
                "nam_stft",
                "checkpoint_esr",
                "checkpoint_stft",
                "best_checkpoint",
                "outdir",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(f"[done] wrote {summary_path} and {csv_path}")
    if results:
        best = results[0]
        print(
            f"[best] {best['name']} nam_esr={best['nam_esr']:.6f} "
            f"checkpoint_esr={best['checkpoint_esr']:.6f}"
        )


if __name__ == "__main__":
    main()
