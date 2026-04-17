import numpy as np

from nam import data_alignment


def test_compute_alignment_diagnostics_reports_expected_delay():
    n = 16384
    dry = np.zeros((1, n), dtype=np.float32)
    dry[:, 2000:2300] = 1.0
    wet = np.zeros((1, n), dtype=np.float32)
    wet[:, 2024:2324] = 1.0
    diag = data_alignment.compute_alignment_diagnostics(
        dry=dry,
        wet=wet,
        max_delay_samples=128,
        analysis_samples=8192,
        residual_windows=4,
        residual_window_samples=2048,
    )
    assert int(diag["global_delay_samples"]) == 24
    assert float(diag["global_peak_ratio"]) > 1.0
    assert "residual_delay_std_samples" in diag


def test_preprocess_pair_reports_quality_gate_failures_for_clipping():
    n = 32768
    dry = np.random.randn(1, n).astype(np.float32) * 0.1
    wet = np.ones((1, n), dtype=np.float32)
    _, _, report = data_alignment.preprocess_pair(
        dry=dry,
        wet=wet,
        max_delay_samples=0,
        alignment_mode="none",
        normalization_mode="none",
        clip_threshold=0.999,
        max_clip_fraction=0.001,
    )
    assert report["quality"]["passed"] is False
    assert any("clip_fraction" in m for m in report["quality"]["messages"])


def test_piecewise_alignment_stays_near_global_delay_when_constrained():
    n = 32768
    delay = 120
    dry = np.zeros((1, n), dtype=np.float32)
    dry[:, 4000:4500] = 1.0
    wet = np.zeros((1, n), dtype=np.float32)
    wet[:, 4000 + delay : 4500 + delay] = 1.0
    _, _, report = data_alignment.preprocess_pair(
        dry=dry,
        wet=wet,
        max_delay_samples=512,
        alignment_mode="piecewise",
        piecewise_block_samples=8192,
        piecewise_smooth_blocks=3,
        piecewise_max_residual_delay_samples=32,
        piecewise_min_peak_ratio=1.02,
        normalization_mode="none",
    )
    assert report["alignment"]["piecewise_applied"] is True
    assert abs(float(report["alignment"]["piecewise_delay_min"]) - delay) <= 64
    assert abs(float(report["alignment"]["piecewise_delay_max"]) - delay) <= 64

