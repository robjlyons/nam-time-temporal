"""
Alignment and normalization utilities for long temporal captures.
"""

from typing import Literal as _Literal
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
from scipy import signal as _signal

from .data_streaming import apply_delay_alignment
from .data_streaming import estimate_delay_samples


def _corr_delay_and_ratio(
    dry_1d: _np.ndarray,
    wet_1d: _np.ndarray,
    max_delay_samples: int,
) -> _Tuple[int, float]:
    corr = _signal.correlate(wet_1d, dry_1d, mode="full", method="fft")
    center = len(corr) // 2
    lo = center - int(max_delay_samples)
    hi = center + int(max_delay_samples) + 1
    view = corr[lo:hi]
    idx = int(_np.argmax(view))
    peak = float(view[idx])
    abs_view = _np.abs(view).copy()
    abs_view[idx] = 0.0
    next_peak = float(abs_view.max()) if abs_view.size > 0 else 0.0
    ratio = abs(peak) / (abs(next_peak) + 1e-12)
    delay = int(idx + lo - center)
    return delay, float(ratio)


def estimate_delay_with_confidence(
    dry: _np.ndarray,
    wet: _np.ndarray,
    max_delay_samples: int = 96_000,
    analysis_samples: int = 262_144,
) -> _Tuple[int, float]:
    x = dry[0] if dry.ndim == 2 else dry
    y = wet[0] if wet.ndim == 2 else wet
    n = min(len(x), len(y), int(analysis_samples))
    if n <= 16:
        return 0, 1.0
    x = x[:n]
    y = y[:n]
    return _corr_delay_and_ratio(x, y, max_delay_samples=int(max_delay_samples))


def _piecewise_delays(
    dry: _np.ndarray,
    wet: _np.ndarray,
    max_delay_samples: int,
    block_samples: int,
    hop_samples: int,
    smooth_blocks: int,
    min_peak_ratio: float = 1.0,
) -> _Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    x = dry[0] if dry.ndim == 2 else dry
    y = wet[0] if wet.ndim == 2 else wet
    n = min(len(x), len(y))
    bs = max(256, int(block_samples))
    hs = max(1, int(hop_samples))
    starts = list(range(0, max(1, n - bs), hs))
    if not starts:
        starts = [0]
    centers = []
    delays = []
    ratios = []
    prev_delay = 0.0
    for s in starts:
        e = min(n, s + bs)
        if e - s < 64:
            continue
        d, r = _corr_delay_and_ratio(x[s:e], y[s:e], max_delay_samples=max_delay_samples)
        if r < float(min_peak_ratio):
            d = int(round(prev_delay))
        prev_delay = float(d)
        centers.append(s + (e - s) // 2)
        delays.append(d)
        ratios.append(float(r))
    if not centers:
        centers = [0, n - 1]
        delays = [0, 0]
        ratios = [1.0, 1.0]
    centers_arr = _np.asarray(centers, dtype=_np.float64)
    delays_arr = _np.asarray(delays, dtype=_np.float64)
    if smooth_blocks > 1 and len(delays_arr) >= smooth_blocks:
        k = int(smooth_blocks)
        kernel = _np.ones((k,), dtype=_np.float64) / float(k)
        delays_arr = _np.convolve(delays_arr, kernel, mode="same")
    return centers_arr, delays_arr, _np.asarray(ratios, dtype=_np.float64)


def apply_piecewise_delay_alignment(
    dry: _np.ndarray,
    wet: _np.ndarray,
    max_delay_samples: int = 96_000,
    block_samples: int = 65_536,
    hop_samples: _Optional[int] = None,
    smooth_blocks: int = 3,
    max_residual_delay_samples: int = 512,
    min_peak_ratio: float = 1.02,
) -> _Tuple[_np.ndarray, _np.ndarray, dict]:
    n = min(dry.shape[1], wet.shape[1])
    x = dry[:, :n]
    y = wet[:, :n]
    global_delay, global_ratio = estimate_delay_with_confidence(
        dry=x, wet=y, max_delay_samples=max_delay_samples
    )
    x, y = apply_delay_alignment(dry=x, wet=y, delay_samples=int(global_delay))
    n = min(x.shape[1], y.shape[1])
    hop = int(hop_samples) if hop_samples is not None else max(1, block_samples // 2)
    centers, delays_residual, ratios = _piecewise_delays(
        dry=x,
        wet=y,
        max_delay_samples=int(max_residual_delay_samples),
        block_samples=block_samples,
        hop_samples=hop,
        smooth_blocks=smooth_blocks,
        min_peak_ratio=float(min_peak_ratio),
    )
    sample_idx = _np.arange(n, dtype=_np.float64)
    residual_curve = _np.interp(sample_idx, centers, delays_residual).astype(_np.float64)
    wet_src = sample_idx + residual_curve
    valid = (wet_src >= 0.0) & (wet_src <= (n - 1))
    if valid.sum() < 64:
        return x, y, {
            "piecewise_applied": False,
            "piecewise_reason": "insufficient_valid_samples",
            "piecewise_delay_min": float(global_delay),
            "piecewise_delay_max": float(global_delay),
            "piecewise_delay_std": 0.0,
        }
    out_x = x[:, valid]
    src = wet_src[valid]
    out_y = _np.empty((y.shape[0], out_x.shape[1]), dtype=_np.float32)
    xp = _np.arange(n, dtype=_np.float64)
    for c in range(y.shape[0]):
        out_y[c] = _np.interp(src, xp, y[c].astype(_np.float64)).astype(_np.float32)
    return out_x, out_y, {
        "piecewise_applied": True,
        "global_delay_samples": int(global_delay),
        "global_peak_ratio": float(global_ratio),
        "piecewise_residual_delay_min": float(delays_residual.min()) if len(delays_residual) else 0.0,
        "piecewise_residual_delay_max": float(delays_residual.max()) if len(delays_residual) else 0.0,
        "piecewise_residual_delay_std": float(delays_residual.std()) if len(delays_residual) else 0.0,
        "piecewise_delay_min": float(global_delay + delays_residual.min()) if len(delays_residual) else float(global_delay),
        "piecewise_delay_max": float(global_delay + delays_residual.max()) if len(delays_residual) else float(global_delay),
        "piecewise_delay_std": float(delays_residual.std()) if len(delays_residual) else 0.0,
        "piecewise_peak_ratio_min": float(ratios.min()) if len(ratios) else 1.0,
        "piecewise_peak_ratio_mean": float(ratios.mean()) if len(ratios) else 1.0,
    }


def compute_alignment_diagnostics(
    dry: _np.ndarray,
    wet: _np.ndarray,
    max_delay_samples: int = 96_000,
    analysis_samples: int = 262_144,
    residual_windows: int = 8,
    residual_window_samples: int = 65_536,
) -> dict:
    global_delay, peak_ratio = estimate_delay_with_confidence(
        dry=dry,
        wet=wet,
        max_delay_samples=max_delay_samples,
        analysis_samples=analysis_samples,
    )
    x, y = apply_delay_alignment(dry=dry, wet=wet, delay_samples=global_delay)
    n = min(x.shape[1], y.shape[1])
    if n < 256:
        residual_delays: list[int] = [0]
    else:
        win = max(512, min(int(residual_window_samples), n))
        n_windows = max(1, int(residual_windows))
        starts = _np.linspace(0, max(0, n - win), num=n_windows, dtype=_np.int64)
        residual_delays = []
        for s in starts:
            e = int(s + win)
            max_residual_search = max(
                8, min(int(max_delay_samples), int(win // 32), 512)
            )
            d = estimate_delay_samples(
                dry=x[:, int(s):e],
                wet=y[:, int(s):e],
                max_delay_samples=max_residual_search,
                analysis_samples=win,
            )
            residual_delays.append(int(d))
    arr = _np.asarray(residual_delays, dtype=_np.float64)
    return {
        "global_delay_samples": int(global_delay),
        "global_peak_ratio": float(peak_ratio),
        "residual_delays_samples": [int(v) for v in residual_delays],
        "residual_delay_std_samples": float(arr.std()) if arr.size else 0.0,
        "residual_delay_drift_samples": float(arr.max() - arr.min()) if arr.size else 0.0,
    }


def _fit_affine_gain(
    dry: _np.ndarray,
    wet: _np.ndarray,
) -> _Tuple[float, float]:
    x = dry.mean(axis=0).astype(_np.float64)
    y = wet.mean(axis=0).astype(_np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_var = float(_np.mean((x - x_mean) ** 2))
    if x_var <= 1e-12:
        return 1.0, 0.0
    cov = float(_np.mean((x - x_mean) * (y - y_mean)))
    a = cov / x_var
    b = y_mean - a * x_mean
    return float(a), float(b)


def preprocess_pair(
    dry: _np.ndarray,
    wet: _np.ndarray,
    max_delay_samples: int = 96_000,
    analysis_samples: int = 262_144,
    alignment_mode: _Literal["none", "global", "piecewise"] = "global",
    piecewise_block_samples: int = 65_536,
    piecewise_hop_samples: _Optional[int] = None,
    piecewise_smooth_blocks: int = 3,
    piecewise_max_residual_delay_samples: int = 512,
    piecewise_min_peak_ratio: float = 1.02,
    normalization_mode: _Literal["none", "rms_match", "affine"] = "none",
    remove_dc: bool = False,
    min_alignment_peak_ratio: float = 1.25,
    max_residual_delay_std_samples: float = 4.0,
    clip_threshold: float = 0.999,
    max_clip_fraction: float = 0.02,
) -> _Tuple[_np.ndarray, _np.ndarray, dict]:
    x = dry.astype(_np.float32, copy=False)
    y = wet.astype(_np.float32, copy=False)
    diag = compute_alignment_diagnostics(
        dry=x,
        wet=y,
        max_delay_samples=max_delay_samples,
        analysis_samples=analysis_samples,
    )
    alignment_report = {"mode": alignment_mode}
    if alignment_mode == "none":
        pass
    elif alignment_mode == "global":
        x, y = apply_delay_alignment(
            dry=x, wet=y, delay_samples=int(diag["global_delay_samples"])
        )
    elif alignment_mode == "piecewise":
        x, y, piecewise_report = apply_piecewise_delay_alignment(
            dry=x,
            wet=y,
            max_delay_samples=max_delay_samples,
            block_samples=piecewise_block_samples,
            hop_samples=piecewise_hop_samples,
            smooth_blocks=piecewise_smooth_blocks,
            max_residual_delay_samples=piecewise_max_residual_delay_samples,
            min_peak_ratio=piecewise_min_peak_ratio,
        )
        alignment_report.update(piecewise_report)
    else:
        raise ValueError(f"Unsupported alignment_mode={alignment_mode}")

    norm_report = {"mode": normalization_mode, "remove_dc": bool(remove_dc)}
    if remove_dc:
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
    if normalization_mode == "none":
        pass
    elif normalization_mode == "rms_match":
        dry_rms = float(_np.sqrt(_np.mean(x.astype(_np.float64) ** 2) + 1e-12))
        wet_rms = float(_np.sqrt(_np.mean(y.astype(_np.float64) ** 2) + 1e-12))
        scale = wet_rms / max(dry_rms, 1e-12)
        x = (x * scale).astype(_np.float32)
        norm_report.update({"scale": float(scale), "dry_rms": dry_rms, "wet_rms": wet_rms})
    elif normalization_mode == "affine":
        a, b = _fit_affine_gain(x, y)
        x = (x * a).astype(_np.float32)
        if remove_dc:
            y = (y - b).astype(_np.float32)
        norm_report.update({"gain_a": float(a), "offset_b": float(b)})
    else:
        raise ValueError(f"Unsupported normalization_mode={normalization_mode}")

    clip_dry = float(_np.mean(_np.abs(x) >= float(clip_threshold)))
    clip_wet = float(_np.mean(_np.abs(y) >= float(clip_threshold)))
    quality_messages = []
    if float(diag["global_peak_ratio"]) < float(min_alignment_peak_ratio):
        quality_messages.append(
            f"low_alignment_peak_ratio:{diag['global_peak_ratio']:.4f}<"
            f"{float(min_alignment_peak_ratio):.4f}"
        )
    if float(diag["residual_delay_std_samples"]) > float(max_residual_delay_std_samples):
        quality_messages.append(
            f"residual_delay_std:{diag['residual_delay_std_samples']:.4f}>"
            f"{float(max_residual_delay_std_samples):.4f}"
        )
    if clip_dry > float(max_clip_fraction):
        quality_messages.append(
            f"dry_clip_fraction:{clip_dry:.6f}>{float(max_clip_fraction):.6f}"
        )
    if clip_wet > float(max_clip_fraction):
        quality_messages.append(
            f"wet_clip_fraction:{clip_wet:.6f}>{float(max_clip_fraction):.6f}"
        )
    quality = {
        "passed": len(quality_messages) == 0,
        "messages": quality_messages,
        "clip_fraction_dry": clip_dry,
        "clip_fraction_wet": clip_wet,
    }
    report = {
        "alignment": {**diag, **alignment_report},
        "normalization": norm_report,
        "quality": quality,
    }
    n = min(x.shape[1], y.shape[1])
    return x[:, :n], y[:, :n], report


def auto_align_pair(
    dry: _np.ndarray, wet: _np.ndarray, max_delay_samples: int = 96_000
) -> _Tuple[_np.ndarray, _np.ndarray, int]:
    delay = estimate_delay_samples(dry=dry, wet=wet, max_delay_samples=max_delay_samples)
    x, y = apply_delay_alignment(dry=dry, wet=wet, delay_samples=delay)
    return x, y, delay
