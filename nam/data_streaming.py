"""
Streaming-friendly dataset helpers for long temporal effects.
"""

from dataclasses import dataclass as _dataclass
import math as _math
from pathlib import Path as _Path
from typing import Literal as _Literal
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
from scipy.io import wavfile as _wavfile
from scipy import signal as _signal
import torch as _torch
from torch.utils.data import Dataset as _Dataset


def _ensure_2d(x: _np.ndarray) -> _np.ndarray:
    return x[None, :] if x.ndim == 1 else x


def _to_float32_audio(x: _np.ndarray) -> _np.ndarray:
    if _np.issubdtype(x.dtype, _np.floating):
        return x.astype(_np.float32)
    if _np.issubdtype(x.dtype, _np.signedinteger):
        info = _np.iinfo(x.dtype)
        scale = float(max(abs(info.min), info.max))
        return x.astype(_np.float32) / scale
    if _np.issubdtype(x.dtype, _np.unsignedinteger):
        info = _np.iinfo(x.dtype)
        midpoint = (info.max + 1) / 2.0
        return (x.astype(_np.float32) - midpoint) / midpoint
    raise ValueError(f"Unsupported audio dtype {x.dtype}")


def _load_wav(path: str | _Path) -> tuple[_np.ndarray, int]:
    sample_rate, x = _wavfile.read(str(path))
    x = _to_float32_audio(_np.asarray(x))
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 2:
        # wavfile returns (N,C), we use (C,N)
        x = x.T
    else:
        raise ValueError(f"Unsupported wav shape {x.shape} for {path}")
    return x.astype(_np.float32), int(sample_rate)


def _resample_channels(
    x: _np.ndarray, orig_sr: int, target_sr: int
) -> _np.ndarray:
    if int(orig_sr) == int(target_sr):
        return x
    g = _math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    y = _signal.resample_poly(x, up=up, down=down, axis=1)
    return y.astype(_np.float32)


@_dataclass
class AudioPair:
    dry: _np.ndarray  # (C,N)
    wet: _np.ndarray  # (C,N)
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.dry.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.dry.shape[1])


def load_audio_pair(
    input_path: str | _Path,
    output_path: str | _Path,
    target_sample_rate: _Optional[int] = None,
    force_mono: bool = False,
) -> AudioPair:
    dry, dry_sr = _load_wav(input_path)
    wet, wet_sr = _load_wav(output_path)
    dry = _ensure_2d(dry)
    wet = _ensure_2d(wet)
    if force_mono:
        dry = dry.mean(axis=0, keepdims=True)
        wet = wet.mean(axis=0, keepdims=True)

    if target_sample_rate is None:
        target_sample_rate = wet_sr if wet_sr != dry_sr else dry_sr
    if dry_sr != target_sample_rate:
        dry = _resample_channels(dry, orig_sr=dry_sr, target_sr=target_sample_rate)
    if wet_sr != target_sample_rate:
        wet = _resample_channels(wet, orig_sr=wet_sr, target_sr=target_sample_rate)
    if dry.shape[0] != wet.shape[0]:
        raise ValueError(
            f"Mismatched channels between dry ({dry.shape[0]}) and wet ({wet.shape[0]})."
        )
    n = min(dry.shape[1], wet.shape[1])
    return AudioPair(dry=dry[:, :n], wet=wet[:, :n], sample_rate=int(target_sample_rate))


def estimate_delay_samples(
    dry: _np.ndarray,
    wet: _np.ndarray,
    max_delay_samples: int = 96_000,
    analysis_samples: int = 262_144,
) -> int:
    """
    Positive delay means wet lags dry.
    """
    x = dry[0] if dry.ndim == 2 else dry
    y = wet[0] if wet.ndim == 2 else wet
    n = min(len(x), len(y), int(analysis_samples))
    x = x[:n]
    y = y[:n]
    corr = _signal.correlate(y, x, mode="full", method="fft")
    center = len(corr) // 2
    lo = center - max_delay_samples
    hi = center + max_delay_samples + 1
    peak = int(_np.argmax(corr[lo:hi])) + lo
    return int(peak - center)


def apply_delay_alignment(
    dry: _np.ndarray,
    wet: _np.ndarray,
    delay_samples: int,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    if delay_samples > 0:
        return dry[:, :-delay_samples], wet[:, delay_samples:]
    if delay_samples < 0:
        d = -delay_samples
        return dry[:, d:], wet[:, :-d]
    return dry, wet


class LongSequenceDataset(_Dataset):
    """
    Random chunk sampler for long aligned dry/wet streams.
    Returns:
    - input_chunk: (C, context+target)
    - target_chunk: (C, target)
    """

    def __init__(
        self,
        pair: AudioPair,
        context_samples: int,
        target_samples: int,
        overlap_samples: int = 0,
        split: _Literal["train", "validation"] = "train",
        validation_fraction: float = 0.1,
        seed: int = 42,
        epoch_steps: int = 10_000,
        deterministic: bool = False,
        active_sampling_ratio: float = 0.0,
        active_rms_quantile: float = 0.8,
        active_window_min_rms: _Optional[float] = None,
        validation_require_active: bool = False,
    ):
        if context_samples < 1 or target_samples < 1:
            raise ValueError("context_samples and target_samples must be positive.")
        self._dry = _torch.from_numpy(pair.dry)
        self._wet = _torch.from_numpy(pair.wet)
        self._context = int(context_samples)
        self._target = int(target_samples)
        self._overlap = int(overlap_samples)
        self._split = split
        self._rng = _np.random.default_rng(seed)
        self._epoch_steps = int(epoch_steps)
        self._deterministic = bool(deterministic)
        self._active_sampling_ratio = float(active_sampling_ratio)
        self._active_rms_quantile = float(active_rms_quantile)
        self._active_window_min_rms = (
            None if active_window_min_rms is None else float(active_window_min_rms)
        )
        self._validation_require_active = bool(validation_require_active)
        self.sample_rate = pair.sample_rate

        n = pair.num_samples
        valid_start = self._context - 1
        valid_stop = n - self._target
        if valid_stop <= valid_start:
            raise ValueError("Audio too short for configured context/target windows.")
        cut = int((valid_stop - valid_start) * (1.0 - validation_fraction)) + valid_start
        if split == "train":
            self._start_min = valid_start
            self._start_max = max(valid_start + 1, cut)
        else:
            self._start_min = max(valid_start, cut)
            self._start_max = valid_stop
        if self._start_max <= self._start_min:
            self._start_max = self._start_min + 1

        all_starts = _np.arange(self._start_min, self._start_max, dtype=_np.int64)
        self._active_starts = self._compute_active_starts(pair, all_starts)

        if self._deterministic:
            # Fixed validation windows reduce val-loss noise across checkpoints.
            start_pool = all_starts
            if self._split == "validation" and self._validation_require_active:
                if self._active_starts is not None and len(self._active_starts) > 0:
                    start_pool = self._active_starts
            if len(start_pool) == 1:
                self._deterministic_starts = _np.full(
                    (self._epoch_steps,), int(start_pool[0]), dtype=_np.int64
                )
            else:
                idx = _np.linspace(
                    0, len(start_pool) - 1, num=self._epoch_steps, endpoint=True
                )
                self._deterministic_starts = start_pool[idx.astype(_np.int64)]
        else:
            self._deterministic_starts = None

    def _compute_active_starts(
        self, pair: AudioPair, starts: _np.ndarray
    ) -> _Optional[_np.ndarray]:
        if len(starts) == 0:
            return None
        need_active = (
            self._active_sampling_ratio > 0.0
            or (
                self._split == "validation"
                and self._validation_require_active
            )
        )
        if not need_active:
            return None
        wet = pair.wet.mean(axis=0).astype(_np.float32, copy=False)
        wet_sq = wet * wet
        csum = _np.concatenate(
            [_np.array([0.0], dtype=_np.float64), _np.cumsum(wet_sq, dtype=_np.float64)]
        )
        win_end = starts + self._target
        win_mean_sq = (csum[win_end] - csum[starts]) / float(self._target)
        win_rms = _np.sqrt(_np.maximum(win_mean_sq, 0.0))
        if self._active_window_min_rms is not None and self._active_window_min_rms > 0:
            threshold = float(self._active_window_min_rms)
        else:
            q = float(_np.clip(self._active_rms_quantile, 0.0, 1.0))
            threshold = float(_np.quantile(win_rms, q))
        active = starts[win_rms >= threshold]
        if len(active) == 0:
            return None
        return active.astype(_np.int64, copy=False)

    def __len__(self) -> int:
        return self._epoch_steps

    def __getitem__(self, idx: int):
        if self._deterministic_starts is not None:
            start = int(self._deterministic_starts[idx % self._epoch_steps])
        else:
            # Random windows with optional overlap jitter.
            choose_active = (
                self._active_starts is not None
                and self._active_sampling_ratio > 0.0
                and float(self._rng.random()) < self._active_sampling_ratio
            )
            if choose_active:
                j = int(self._rng.integers(0, len(self._active_starts)))
                start = int(self._active_starts[j])
            else:
                start = int(self._rng.integers(self._start_min, self._start_max))
        if self._overlap > 0:
            jitter = int(self._rng.integers(-self._overlap, self._overlap + 1))
            start = min(max(start + jitter, self._start_min), self._start_max - 1)
        x0 = start - (self._context - 1)
        x1 = start + self._target
        y0 = start
        y1 = start + self._target
        return self._dry[:, x0:x1], self._wet[:, y0:y1]
