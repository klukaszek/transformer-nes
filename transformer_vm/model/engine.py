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


def _check_ffn_kinds(model, layer_idx):
    """Validate the fusion contract behind ffn_kinds == 2.

    The C++ engine folds kind-2 (linear passthrough/erase) neurons into a
    direct residual map using only their value row, dropping the gate.
    That is exact only if the gate is the constant-one expression, so a
    mislabeled neuron must fail here instead of silently computing the
    wrong activation.
    """
    w = model.ff_in[layer_idx].weight
    ffn = w.shape[0] // 2
    for j, kind in enumerate(model.ffn_kinds[layer_idx]):
        if kind != 2:
            continue
        gate = w[j]
        if gate[model.one_slot] != 1.0 or int((gate != 0).sum()) != 1:
            raise RuntimeError(
                f"L{layer_idx} ffn neuron {j}: kind-2 gate is not the constant one"
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
        self.n_memory_heads = 0
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
            self.n_memory_heads += sum(1 for g in mem_groups if g >= 0)
            _check_ffn_kinds(model, layer_idx)
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

    @property
    def supports_direct_writes(self) -> bool:
        """True iff insert_write reaches at least one memory lookup hull.

        A model without memory-group metadata would accept direct writes
        and drop every one of them on the floor; callers must fall back to
        full WRITE-token forwards instead.
        """
        return self.dtype == "float32" and self.n_memory_heads > 0

    def insert_write(self, addr: int, value: int, pos: int) -> None:
        """Fast-path a WRITE token into the engine's memory lookup hulls."""
        if self.dtype != "float32":
            raise RuntimeError("direct WRITE insertion is currently float32-only")
        if self.n_memory_heads == 0:
            raise RuntimeError(
                "model has no memory-tagged lookup heads; a direct WRITE would "
                "be silently lost -- use a full WRITE-token forward instead"
            )
        self._engine.insert_write_f32(addr, value, pos)
