"""
Alignment utilities for long temporal captures.
"""

from typing import Tuple as _Tuple

import numpy as _np

from .data_streaming import apply_delay_alignment
from .data_streaming import estimate_delay_samples


def auto_align_pair(
    dry: _np.ndarray, wet: _np.ndarray, max_delay_samples: int = 96_000
) -> _Tuple[_np.ndarray, _np.ndarray, int]:
    delay = estimate_delay_samples(dry=dry, wet=wet, max_delay_samples=max_delay_samples)
    x, y = apply_delay_alignment(dry=dry, wet=wet, delay_samples=delay)
    return x, y, delay
