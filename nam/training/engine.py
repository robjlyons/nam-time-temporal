from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..data_alignment import auto_align_pair
from ..data_streaming import LongSequenceDataset
from ..data_streaming import load_audio_pair
from ..train.lightning_module import LightningModule
from .callbacks import AudioPreviewCallback
from .callbacks import MetricsDumperCallback
from .callbacks import StepProgressCallback
from .config import TemporalTrainingConfig


def _collate_channels(batch):
    """
    Flatten channel dimension into batch dimension so mono/stereo data can train
    with mono model interfaces.
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,C,L)
    y = torch.stack(ys, dim=0)  # (B,C,L)
    b, c, lx = x.shape
    _, _, ly = y.shape
    return x.reshape(b * c, lx), y.reshape(b * c, ly)


def _choose_accelerator(device: str):
    if device == "cpu":
        return "cpu", 1
    if device == "gpu":
        return "gpu", 1
    if torch.cuda.is_available():
        return "gpu", 1
    if torch.backends.mps.is_available():
        return "gpu", 1
    return "cpu", 1


def train_temporal_model(config: TemporalTrainingConfig):
    config.outdir.mkdir(parents=True, exist_ok=True)
    print("[progress] loading audio pair...")
    pair = load_audio_pair(
        input_path=config.input_wav,
        output_path=config.output_wav,
        target_sample_rate=None,
        force_mono=config.force_mono,
    )
    print(
        f"[progress] loaded audio | sample_rate={pair.sample_rate} "
        f"channels={pair.channels} samples={pair.num_samples}"
    )
    if config.align_max_delay_samples is not None:
        print("[progress] estimating delay and aligning...")
        dry, wet, delay = auto_align_pair(
            dry=pair.dry, wet=pair.wet, max_delay_samples=config.align_max_delay_samples
        )
        pair.dry = dry
        pair.wet = wet
        (config.outdir / "alignment.txt").write_text(f"estimated_delay_samples={delay}\n")
        print(f"[progress] alignment complete | estimated_delay_samples={delay}")

    train_ds = LongSequenceDataset(
        pair=pair,
        context_samples=config.context_samples,
        target_samples=config.target_samples,
        overlap_samples=config.overlap_samples,
        split="train",
        validation_fraction=config.validation_fraction,
        epoch_steps=config.epoch_steps,
    )
    val_ds = LongSequenceDataset(
        pair=pair,
        context_samples=config.context_samples,
        target_samples=config.target_samples,
        overlap_samples=0,
        split="validation",
        validation_fraction=config.validation_fraction,
        epoch_steps=max(128, config.epoch_steps // 10),
    )
    dl_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "collate_fn": _collate_channels,
    }
    if config.num_workers > 0:
        dl_kwargs["persistent_workers"] = config.persistent_workers
        dl_kwargs["prefetch_factor"] = config.prefetch_factor
    train_dl = DataLoader(train_ds, **dl_kwargs)
    val_dl = DataLoader(val_ds, **dl_kwargs)

    n_train_batches = len(train_dl)
    val_check_interval = int(config.val_check_interval)
    if val_check_interval > n_train_batches:
        print(
            f"[progress] val_check_interval={val_check_interval} is greater than "
            f"train batches per epoch ({n_train_batches}); using {n_train_batches}. "
            "Raise epoch_steps (roughly val_check_interval * batch_size or higher) "
            "if you want less frequent validation."
        )
        val_check_interval = n_train_batches

    module_config = {
        "net": {
            "name": "TemporalHybrid",
            "config": {
                "hidden_size": config.hidden_size,
                "local_channels": config.local_channels,
                "local_kernel_size": config.local_kernel_size,
                "local_layers": config.local_layers,
                "fuse_alpha": 1.0,
                "context_samples": config.context_samples,
                "sample_rate": pair.sample_rate,
            },
        },
        "loss": {
            "mse_weight": 1.0,
            "mrstft_weight": config.mrstft_weight,
            "val_loss": "esr",
        },
        "optimizer": {"lr": config.learning_rate},
        "lr_scheduler": None,
    }
    if config.resume is not None:
        module_config["checkpoint_path"] = str(config.resume)
    model = LightningModule.init_from_config(module_config)
    model.net.sample_rate = pair.sample_rate

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(config.outdir / "checkpoints"),
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        every_n_train_steps=config.checkpoint_every_n_steps,
        filename="step{step:08d}-val{val_loss:.5f}",
        auto_insert_metric_name=False,
    )
    callbacks = [
        ckpt_callback,
        AudioPreviewCallback(
            outdir=config.outdir,
            every_n_steps=config.preview_every_n_steps,
            sample_rate=pair.sample_rate,
        ),
        MetricsDumperCallback(outdir=config.outdir),
        StepProgressCallback(every_n_steps=max(10, config.checkpoint_every_n_steps // 5)),
    ]

    accelerator, devices = _choose_accelerator(config.device)
    trainer = pl.Trainer(
        default_root_dir=str(config.outdir),
        callbacks=callbacks,
        max_steps=config.max_steps,
        val_check_interval=val_check_interval,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
    )
    trainer.fit(model, train_dl, val_dl, ckpt_path=str(config.resume) if config.resume else None)
    best_ckpt = ckpt_callback.best_model_path or ckpt_callback.last_model_path
    if best_ckpt:
        model = LightningModule.load_from_checkpoint(
            best_ckpt, **LightningModule.parse_config(module_config)
        )
        model.net.sample_rate = pair.sample_rate
        model.net.export(config.outdir, basename="model")
    return model
