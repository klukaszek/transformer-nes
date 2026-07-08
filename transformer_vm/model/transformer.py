"""VanillaTransformer with ReGLU FFN: the container the analytic weights
populate.  Inference runs in the generated C++ engine, not here."""

import math

import torch
import torch.nn as nn

# Analytical weights are constructed and serialized in float64 for stable
# coefficient storage.  The 6502 runtime defaults to the C++ engine's float32
# forward path after export.
torch.set_default_dtype(torch.float64)


def add_position_encoding(x, pos):
    """Add deterministic position features to the residual stream."""
    x[0] += pos
    x[1] += 1.0 / math.log(2) - 1.0 / math.log(pos + 2)
    x[2] += pos * pos


class VanillaTransformer(nn.Module):
    def __init__(self, vocab, d_model=36, n_heads=18, n_layers=7, d_ffn=36, stop_token_id=0):
        super().__init__()
        self.stop_token_id = stop_token_id
        self.tok = nn.Embedding(vocab, d_model)

        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.ff_in = nn.ModuleList(
            [nn.Linear(d_model, 2 * d_ffn, bias=False) for _ in range(n_layers)]
        )
        self.ff_out = nn.ModuleList(
            [nn.Linear(d_ffn, d_model, bias=False) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab, bias=False)
