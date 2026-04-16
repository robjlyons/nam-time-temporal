# Temporal Validation Playbook

Use this protocol to compare experiments consistently.

## Canonical offline evaluation

- Always evaluate with `evaluate.py` on the same input/target pair.
- Keep `--chunk-size 16384` fixed.
- Evaluate both:
  - best checkpoint (`eval_best_ckpt`)
  - exported model (`eval_model_nam`)

Command template:

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint_or_model.nam \
  --input /path/to/input.wav \
  --target /path/to/output.wav \
  --chunk-size 16384 \
  --outdir /path/to/eval_out
```

## Checkpoint selection rule

1. Prefer checkpoint files named `step########-val#####.ckpt` with the lowest `val`.
2. If none exist, use `checkpoints/last.ckpt`.
3. Keep this rule unchanged for every experiment.

## Sweep runner

Use `tools/temporal_validation_sweep.py` to enforce the same train/eval pipeline:

```bash
python tools/temporal_validation_sweep.py \
  --input ../input.wav \
  --target ../output.wav \
  --run-root runs/temporal_sweep_01 \
  --device auto
```

Outputs:

- `runs/.../leaderboard.json`
- `runs/.../leaderboard.csv`
- per-run `experiment_result.json`

## Promotion runs

After selecting top configs from `leaderboard.csv`, run longer training (30k-60k) with the
same eval protocol, and stop after 2-3 full-file evals with no improvement.
