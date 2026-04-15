"""
Hybrid temporal model for long-memory effects.

This module combines a local causal convolution branch with an LSTM branch and
fuses both outputs. For plugin compatibility, export is constrained to an
existing architecture (`LSTM` by default).
"""

from typing import Literal as _Literal
from typing import Optional as _Optional

import numpy as _np
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

from ._abc import ImportsWeights as _ImportsWeights
from .base import BaseNet as _BaseNet
from .recurrent import LSTM as _LSTM


class TemporalHybrid(_BaseNet, _ImportsWeights):
    """
    Long-memory hybrid model:
    - local branch: shallow causal 1D conv stack
    - temporal branch: LSTM consuming [dry, local] features
    - fusion: weighted sum of local and temporal outputs

    Export intentionally emits a supported architecture to remain compatible
    with NAM plugin loaders.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        local_channels: int = 16,
        local_kernel_size: int = 5,
        local_layers: int = 2,
        fuse_alpha: float = 1.0,
        context_samples: int = 8192,
        export_as: _Literal["LSTM"] = "LSTM",
        train_burn_in: _Optional[int] = None,
        train_truncate: _Optional[int] = None,
        sample_rate: _Optional[float] = None,
        **lstm_kwargs,
    ):
        super().__init__(sample_rate=sample_rate)
        if export_as != "LSTM":
            raise ValueError("TemporalHybrid currently supports export_as='LSTM' only.")
        if export_as == "LSTM" and abs(float(fuse_alpha) - 1.0) > 1e-9:
            raise ValueError(
                "TemporalHybrid export_as='LSTM' requires fuse_alpha=1.0 for "
                "checkpoint/export parity."
            )
        if local_kernel_size < 1 or local_layers < 1:
            raise ValueError("local kernel/layers must be positive")

        self._export_as = export_as
        self._fuse_alpha = float(fuse_alpha)
        self._context_samples = int(context_samples)
        self._local_kernel_size = int(local_kernel_size)
        self._local_layers = int(local_layers)

        local: list[_nn.Module] = []
        in_ch = 1
        for _ in range(local_layers):
            local.append(
                _nn.Conv1d(in_ch, local_channels, kernel_size=local_kernel_size, bias=True)
            )
            local.append(_nn.Tanh())
            in_ch = local_channels
        local.append(_nn.Conv1d(local_channels, 1, kernel_size=1, bias=True))
        self._local = _nn.Sequential(*local)

        self._lstm = _LSTM(
            hidden_size=hidden_size,
            input_size=1,
            train_burn_in=train_burn_in,
            train_truncate=train_truncate,
            sample_rate=sample_rate,
            **lstm_kwargs,
        )

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        return config

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return max(1, self._context_samples)

    def _local_forward(self, x: _torch.Tensor) -> _torch.Tensor:
        # x: (B,L)
        z = x[:, None, :]
        for layer in self._local:
            if isinstance(layer, _nn.Conv1d) and layer.kernel_size[0] > 1:
                pad = layer.kernel_size[0] - 1
                z = _F.pad(z, (pad, 0))
            z = layer(z)
        return z[:, 0, :]

    def _forward(self, x: _torch.Tensor, **kwargs):
        if kwargs:
            raise ValueError("TemporalHybrid does not support extra kwargs")
        local_y = self._local_forward(x)
        temporal_inputs = x[:, :, None]  # (B,L,1)
        temporal_y = self._lstm._forward(temporal_inputs)
        alpha = _torch.tensor(self._fuse_alpha, device=x.device, dtype=x.dtype)
        y = alpha * temporal_y + (1.0 - alpha) * local_y
        return y[:, self.receptive_field - 1 :]

    def _get_export_architecture(self) -> str:
        return self._export_as

    def _export_config(self):
        # Keep exported config compatible by delegating to LSTM config.
        return self._lstm._export_config()

    def _export_weights(self) -> _np.ndarray:
        # Keep exported weight format compatible with LSTM parser.
        burn_in = getattr(self._lstm, "_get_initial_state_burn_in", 48_000)
        device = self._lstm.input_device
        z = _torch.zeros((1, burn_in, 1), device=device)
        h, c = self._lstm._get_initial_state(inputs=z)
        return _np.concatenate(
            [
                self._lstm._export_cell_weights(i, hi, ci)
                for i, (hi, ci) in enumerate(zip(h, c))
            ]
            + [self._lstm._head.export_weights()]
        )

    def import_weights(self, weights):
        self._lstm.import_weights(weights)
