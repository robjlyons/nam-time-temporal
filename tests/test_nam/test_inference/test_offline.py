import torch

from nam.inference.offline import assert_chunk_parity
from nam.models.recurrent import LSTM


def test_chunk_parity_for_lstm():
    model = LSTM(hidden_size=4)
    model.eval()
    x = torch.randn(1, 4096)
    assert_chunk_parity(model, x, chunk_size=512, atol=5e-4)
