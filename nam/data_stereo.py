"""
Stereo/mono helpers for audio pair datasets.
"""

import numpy as _np


def to_mono(x: _np.ndarray) -> _np.ndarray:
    if x.ndim == 1:
        return x[None, :]
    if x.shape[0] == 1:
        return x
    return x.mean(axis=0, keepdims=True)


def assert_compatible_channels(dry: _np.ndarray, wet: _np.ndarray):
    if dry.ndim != 2 or wet.ndim != 2:
        raise ValueError("Expected arrays with shape (channels, samples)")
    if dry.shape[0] != wet.shape[0]:
        raise ValueError(
            f"Dry/wet channel mismatch: {dry.shape[0]} versus {wet.shape[0]}"
        )
