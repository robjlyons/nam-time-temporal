# Temporal Effects Training

This workflow extends NAM training for time-dependent effects (delay/reverb) while
preserving `.nam` export compatibility.

## 1) Prepare paired captures

- `input.wav`: dry stimulus
- `output.wav`: wet response from hardware/software effect
- Use the same sample rate when possible; if mismatched, the trainer resamples
  and can auto-estimate delay alignment.

## 2) Run temporal training

```bash
python train.py \
  --input /path/to/input.wav \
  --output /path/to/output.wav \
  --outdir runs/large_hall \
  --steps 30000 \
  --batch-size 8 \
  --context 16384 \
  --target 8192 \
  --device auto
```

## 3) Resume from checkpoint

```bash
python train.py \
  --input /path/to/input.wav \
  --output /path/to/output.wav \
  --outdir runs/large_hall_resume \
  --resume runs/large_hall/checkpoints/last.ckpt
```

## 4) Artifacts

- Checkpoints: `outdir/checkpoints/*.ckpt`
- Audio previews (periodic): `outdir/previews/step_*/{input,target,model_output}.wav`
- Exported model: `outdir/model.nam`
