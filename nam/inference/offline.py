import torch

from .stateful import StatefulContext
from .stateful import process_stream_chunk


@torch.no_grad()
def process_full(model, x: torch.Tensor) -> torch.Tensor:
    return model(x, pad_start=False)


@torch.no_grad()
def process_chunked(model, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    ctx = StatefulContext()
    y_parts = []
    for i in range(0, x.shape[-1], chunk_size):
        y_parts.append(process_stream_chunk(model, x[..., i : i + chunk_size], ctx))
    return torch.cat(y_parts, dim=-1)


@torch.no_grad()
def assert_chunk_parity(model, x: torch.Tensor, chunk_size: int, atol: float = 1e-4):
    y_full = process_full(model, x)
    y_chunk = process_chunked(model, x, chunk_size=chunk_size)
    if not torch.allclose(y_full, y_chunk, atol=atol):
        diff = torch.max(torch.abs(y_full - y_chunk)).item()
        raise AssertionError(f"Chunk parity failed (max abs diff={diff})")
