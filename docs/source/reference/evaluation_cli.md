# Evaluation CLI

Use `evaluate.py` to score a checkpoint or exported `.nam` model.

```bash
python evaluate.py \
  --checkpoint runs/large_hall/checkpoints/last.ckpt \
  --input test_input.wav \
  --target test_target.wav \
  --outdir eval_large_hall
```

Outputs:

- `metrics.json` with ESR and STFT metrics
- `input.wav`
- `target.wav`
- `model_output.wav`
