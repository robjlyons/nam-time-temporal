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


def test_long_sequence_dataset_active_sampling_biases_to_high_energy_regions():
    n = 24000
    dry = np.random.randn(1, n).astype(np.float32) * 0.01
    wet = np.zeros((1, n), dtype=np.float32)
    wet[:, : n // 2] = np.random.randn(1, n // 2).astype(np.float32) * 1e-4
    wet[:, n // 2 :] = np.random.randn(1, n // 2).astype(np.float32) * 0.1
    pair = AudioPair(dry=dry, wet=wet, sample_rate=48000)
    ds = LongSequenceDataset(
        pair=pair,
        context_samples=256,
        target_samples=512,
        split="train",
        epoch_steps=32,
        active_sampling_ratio=1.0,
        active_rms_quantile=0.8,
    )
    energies = []
    for i in range(16):
        _, y = ds[i]
        energies.append(float((y**2).mean().item()))
    assert np.mean(energies) > 1e-3
