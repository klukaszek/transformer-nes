"""Python glue for the sparse C++ weights engine (weights_engine.cpp).

Exports a built VanillaTransformer's layers as CSR arrays into the
engine; forward() then runs one token position through the entire layer
stack in C++ -- one Python/pybind crossing per position instead of six
per layer.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

_ATTENTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention"
)
_module = None


def _load_module():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_engine.cpp")
        _module = load(
            name="weights_engine",
            sources=[src],
            extra_cflags=["-O2", "-std=c++17"],
            extra_include_paths=[_ATTENTION_DIR],
            verbose=False,
        )
    return _module


def _csr(tensor):
    m = sp.csr_matrix(tensor.numpy())
    return (
        m.indptr.astype(np.int32),
        m.indices.astype(np.int32),
        m.data.astype(np.float64),
    )


class SparseWeightsEngine:
    """The whole layer stack per call; hulls for lookup heads live in C++."""

    def __init__(self, model, *, dtype: str = "float32"):
        mod = _load_module()
        d_model = model.tok.weight.shape[1]
        n_heads = model.attn[0].num_heads
        if dtype not in ("float64", "float32"):
            raise ValueError(f"unknown dtype {dtype!r}")
        self.dtype = dtype
        self._engine = mod.WeightsEngine(d_model, n_heads)
        for layer_idx in range(len(model.attn)):
            cache_groups = getattr(model, "head_cache_groups", None)
            if cache_groups is None:
                groups = [
                    h if kind == 1 else -1
                    for h, kind in enumerate(model.head_kinds[layer_idx])
                ]
            else:
                groups = cache_groups[layer_idx]
            memory_groups = getattr(model, "head_memory_groups", None)
            if memory_groups is None:
                mem_groups = [-1] * n_heads
            else:
                mem_groups = memory_groups[layer_idx]
            self._engine.add_layer(
                *_csr(model.attn[layer_idx].in_proj_weight),
                *_csr(model.attn[layer_idx].out_proj.weight),
                *_csr(model.ff_in[layer_idx].weight),
                *_csr(model.ff_out[layer_idx].weight),
                model.ff_out[layer_idx].weight.shape[1],
                np.asarray(model.head_kinds[layer_idx], dtype=np.int32),
                np.asarray(model.head_tiebreak[layer_idx], dtype=np.int32),
                np.asarray(model.ffn_kinds[layer_idx], dtype=np.int32),
                np.asarray(groups, dtype=np.int32),
                np.asarray(mem_groups, dtype=np.int32),
            )

    def reset(self):
        self._engine.reset()

    def forward(self, x: np.ndarray) -> None:
        """Run one position through all layers, updating x in place.

        x must be a contiguous float64/float32 array matching this engine's
        dtype; a pybind cast would operate on a copy and drop the updates.
        """
        if self.dtype == "float32":
            self._engine.forward_f32(x)
        else:
            self._engine.forward(x)

    def insert_write(self, addr: int, value: int, pos: int) -> None:
        """Fast-path a WRITE token into the engine's memory lookup hulls."""
        if self.dtype != "float32":
            raise RuntimeError("direct WRITE insertion is currently float32-only")
        self._engine.insert_write_f32(addr, value, pos)
