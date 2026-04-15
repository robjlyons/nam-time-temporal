from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import torch

from ..models.recurrent import LSTM
from ..models.temporal_hybrid import TemporalHybrid

HiddenState = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class StatefulContext:
    hidden_state: Optional[HiddenState] = None

    def reset(self):
        self.hidden_state = None


def _resolve_lstm(model):
    if isinstance(model, LSTM):
        return model
    if isinstance(model, TemporalHybrid):
        return model._lstm
    return None


@torch.no_grad()
def process_stream_chunk(
    model, chunk: torch.Tensor, context: Optional[StatefulContext] = None
) -> torch.Tensor:
    """
    Process one chunk while carrying hidden state when model supports it.
    chunk: (L,) or (B,L)
    """
    lstm = _resolve_lstm(model)
    if lstm is None:
        return model(chunk, pad_start=False)
    if context is None:
        context = StatefulContext()
    x = chunk if chunk.ndim == 2 else chunk[None]
    if isinstance(model, TemporalHybrid):
        local_y = model._local_forward(x)
        x2 = x[:, :, None]
        out_features, hidden = lstm._core(
            x2, context.hidden_state or lstm._initial_state(len(x))
        )
        context.hidden_state = hidden
        out = lstm._apply_head(out_features)
        alpha = torch.tensor(model._fuse_alpha, device=out.device, dtype=out.dtype)
        out = alpha * out + (1.0 - alpha) * local_y
    else:
        x2 = x[:, :, None] if x.ndim == 2 else x
        out_features, hidden = lstm._core(x2, context.hidden_state or lstm._initial_state(len(x)))
        context.hidden_state = hidden
        out = lstm._apply_head(out_features)
    return out[0] if chunk.ndim == 1 else out
