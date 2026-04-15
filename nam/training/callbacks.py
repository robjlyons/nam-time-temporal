from pathlib import Path
import time

import numpy as np
import pytorch_lightning as pl
import torch
import wavio


class AudioPreviewCallback(pl.Callback):
    def __init__(self, outdir: Path, every_n_steps: int = 1000, sample_rate: int = 48000):
        super().__init__()
        self._outdir = Path(outdir)
        self._every_n_steps = int(every_n_steps)
        self._sample_rate = int(sample_rate)

    def _write_wav(self, x: np.ndarray, path: Path):
        y = np.clip(x, -1.0, 1.0)
        wavio.write(str(path), (y * (2**15 - 1)).astype(np.int16), self._sample_rate)

    def _emit_preview(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            return
        if isinstance(val_loader, list):
            val_loader = val_loader[0]
        batch = next(iter(val_loader))
        x = batch[0].to(pl_module.device)
        y = batch[-1].to(pl_module.device)
        with torch.no_grad():
            y_hat = pl_module(x, pad_start=False)
        output_dir = self._outdir / "previews" / f"step_{trainer.global_step:08d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_wav(x[0].detach().cpu().numpy(), output_dir / "input.wav")
        self._write_wav(y[0].detach().cpu().numpy(), output_dir / "target.wav")
        self._write_wav(y_hat[0].detach().cpu().numpy(), output_dir / "model_output.wav")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._every_n_steps <= 0:
            return
        if trainer.global_step == 0:
            return
        if trainer.global_step % self._every_n_steps != 0:
            return
        self._emit_preview(trainer, pl_module)


class MetricsDumperCallback(pl.Callback):
    def __init__(self, outdir: Path):
        super().__init__()
        self._path = Path(outdir) / "metrics.csv"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("step,train_loss,val_loss\n")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = trainer.callback_metrics
        train_loss = m.get("train_loss")
        val_loss = m.get("val_loss")
        with self._path.open("a") as fp:
            fp.write(
                f"{trainer.global_step},{'' if train_loss is None else float(train_loss)},{'' if val_loss is None else float(val_loss)}\n"
            )


class StepProgressCallback(pl.Callback):
    """
    Print lightweight progress updates for long runs so users can see training
    advancing even when a dynamic progress bar is not visible.
    """

    def __init__(self, every_n_steps: int = 25):
        super().__init__()
        self._every_n_steps = int(every_n_steps)
        self._t0: float | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._t0 = time.time()
        max_steps = trainer.max_steps
        print(
            f"[progress] training started | max_steps={max_steps} | "
            f"device={trainer.accelerator.__class__.__name__}"
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        step = trainer.global_step
        if step <= 0 or (step % self._every_n_steps != 0 and step != trainer.max_steps):
            return
        max_steps = max(1, int(trainer.max_steps))
        elapsed = 0.0 if self._t0 is None else time.time() - self._t0
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (max_steps - step) / rate if rate > 1e-9 else float("inf")
        pct = 100.0 * step / max_steps
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        train_str = f"{float(train_loss):.6f}" if train_loss is not None else "n/a"
        val_str = f"{float(val_loss):.6f}" if val_loss is not None else "n/a"
        eta_str = f"{eta/60.0:.1f}m" if eta != float("inf") else "n/a"
        print(
            f"[progress] step={step}/{max_steps} ({pct:.1f}%) "
            f"train_loss={train_str} val_loss={val_str} eta={eta_str}"
        )
