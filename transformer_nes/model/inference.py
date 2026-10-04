"""Exact reference inference for a constructed WeightModel.

A plain transformer forward, one token at a time, in float64: the input
slots form the embedding, then each layer applies attention (in its
argmax limit, over every token so far including this one; distinct keys
never tie, and of equal keys the latest counts) and a ReGLU feed-forward
block, both residual.  No position encoding.

Two exact reductions keep it fast without changing any output:

- A head whose query and key rows are the constant ``one`` scores every
  token equally, so its winner is the current token (the passthrough and
  erase heads the construction packs; ``head_kinds`` marks them).
- A cached key that is equal to a newer one can never win again (equal
  scores go to the later token), so it is replaced in place.  Memory
  heads therefore hold one entry per written address.

This is the reference the generated C++ engines are fast versions of,
and it runs any graph built with model/weights.py.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from transformer_nes.model.weights import HARD_K, WeightModel


class _HeadCache:
    """Keys (n, 2) and values (n, 2) of one lookup head, deduplicated."""

    __slots__ = ("keys", "values", "positions", "size", "slot_of_key")

    def __init__(self):
        self.keys = np.zeros((64, 2))
        self.values = np.zeros((64, 2))
        self.positions = np.zeros(64, dtype=np.int64)
        self.size = 0
        self.slot_of_key: dict[tuple[float, float], int] = {}

    def insert(self, key: np.ndarray, value: np.ndarray, position: int) -> None:
        tag = (float(key[0]), float(key[1]))
        slot = self.slot_of_key.get(tag)
        if slot is None:
            if self.size == len(self.keys):
                self.keys = np.vstack([self.keys, np.zeros_like(self.keys)])
                self.values = np.vstack([self.values, np.zeros_like(self.values)])
                self.positions = np.concatenate([self.positions, np.zeros_like(self.positions)])
            slot = self.size
            self.size += 1
            self.slot_of_key[tag] = slot
            self.keys[slot] = key
        self.values[slot] = value
        self.positions[slot] = position

    def attend(self, query: np.ndarray) -> np.ndarray:
        # Distinct keys can still tie on score; the later token wins.
        scores = self.keys[: self.size] @ query
        ties = np.flatnonzero(scores == scores.max())
        winner = ties[np.argmax(self.positions[ties])]
        return self.values[winner]


class ExactInference:
    def __init__(self, model: WeightModel, d_model: int):
        self.d_model = d_model
        self.layers = []
        q_scale = HARD_K * math.sqrt(2.0)
        for li, attn in enumerate(model.attn):
            n_heads = attn.num_heads
            ip = attn.in_proj_weight
            lookup_heads = [h for h in range(n_heads) if model.head_kinds[li][h] == 1]
            self_heads = [h for h in range(n_heads) if model.head_kinds[li][h] == 2]

            def rows(base, heads):
                return [base + 2 * h + c for h in heads for c in (0, 1)]

            layer = {
                "lookup_heads": lookup_heads,
                "q": sp.csr_matrix(ip[rows(0, lookup_heads)] / q_scale),
                "k": sp.csr_matrix(ip[rows(2 * n_heads, lookup_heads)]),
                "v": sp.csr_matrix(ip[rows(4 * n_heads, lookup_heads)]),
                "out_lookup": sp.csr_matrix(attn.out_proj.weight[:, rows(0, lookup_heads)]),
                # A self-attending head returns its own value vector.
                "self_path": sp.csr_matrix(
                    attn.out_proj.weight[:, rows(0, self_heads)] @ ip[rows(4 * n_heads, self_heads)]
                ),
                "ff_in": sp.csr_matrix(model.ff_in[li].weight),
                "ff_out": sp.csr_matrix(model.ff_out[li].weight),
                "d_ffn": model.ff_out[li].weight.shape[1],
                "caches": [_HeadCache() for _ in lookup_heads],
            }
            self.layers.append(layer)
        self.position = -1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run one token whose input slots are set in x (length d_model);
        returns its final residual stream."""
        self.position += 1
        x = np.array(x, dtype=np.float64)
        for layer in self.layers:
            delta = layer["self_path"] @ x
            if layer["lookup_heads"]:
                queries = layer["q"] @ x
                keys = layer["k"] @ x
                values = layer["v"] @ x
                heads_out = np.empty(len(queries))
                for i, cache in enumerate(layer["caches"]):
                    cache.insert(keys[2 * i : 2 * i + 2], values[2 * i : 2 * i + 2], self.position)
                    heads_out[2 * i : 2 * i + 2] = cache.attend(queries[2 * i : 2 * i + 2])
                delta = delta + layer["out_lookup"] @ heads_out
            x = x + delta
            d = layer["d_ffn"]
            pre = layer["ff_in"] @ x
            x = x + layer["ff_out"] @ (np.maximum(pre[:d], 0.0) * pre[d:])
        return x
