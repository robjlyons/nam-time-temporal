import numpy as np

from nam.data_alignment import auto_align_pair
from nam.data_streaming import AudioPair
from nam.data_streaming import LongSequenceDataset


def test_auto_align_pair_detects_positive_delay():
    n = 4096
    dry = np.zeros((1, n), dtype=np.float32)
    dry[:, 200:260] = 1.0
    wet = np.zeros((1, n), dtype=np.float32)
    wet[:, 212:272] = 1.0
    x, y, d = auto_align_pair(dry, wet, max_delay_samples=128)
    assert d == 12
    assert x.shape == y.shape


def test_long_sequence_dataset_shapes():
    n = 22050
    pair = AudioPair(
        dry=np.random.randn(2, n).astype(np.float32) * 0.01,
        wet=np.random.randn(2, n).astype(np.float32) * 0.01,
        sample_rate=44100,
    )
    ds = LongSequenceDataset(
        pair=pair,
        context_samples=256,
        target_samples=512,
        overlap_samples=16,
        split="train",
        epoch_steps=10,
    )
    x, y = ds[0]
    assert x.shape == (2, 256 + 512 - 1)
    assert y.shape == (2, 512)
