from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TemporalTrainingConfig:
    input_wav: Path
    output_wav: Path
    outdir: Path
    batch_size: int = 8
    num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 2
    context_samples: int = 8192
    target_samples: int = 8192
    overlap_samples: int = 1024
    validation_fraction: float = 0.1
    epoch_steps: int = 2000
    max_steps: int = 20000
    val_check_interval: int = 500
    checkpoint_every_n_steps: int = 500
    preview_every_n_steps: int = 1000
    log_every_n_steps: int = 50
    precision: str = "32-true"
    learning_rate: float = 3e-4
    hidden_size: int = 48
    local_channels: int = 16
    local_kernel_size: int = 5
    local_layers: int = 2
    force_mono: bool = False
    align_max_delay_samples: Optional[int] = 96000
    resume: Optional[Path] = None
    device: str = "auto"  # auto|cpu|gpu
    mrstft_weight: Optional[float] = 2e-4
