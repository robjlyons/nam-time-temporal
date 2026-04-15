import torch

from nam.models.temporal_hybrid import TemporalHybrid


def test_temporal_hybrid_forward_and_export():
    model = TemporalHybrid(
        hidden_size=8,
        local_channels=4,
        local_layers=1,
        local_kernel_size=3,
        context_samples=64,
    )
    x = torch.randn(2, 1024)
    y = model(x, pad_start=False)
    assert y.shape == (2, 1024 - 64 + 1)
    cfg = model._export_config()
    weights = model._export_weights()
    assert "hidden_size" in cfg
    assert weights.ndim == 1
