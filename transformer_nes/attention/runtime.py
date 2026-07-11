"""Exact-arithmetic attention runtime for graph evaluators.

HullAttention wraps the O(log n) convex-hull KV cache (hull_ext.cpp /
hull2d_cht.h) for one LookUpDimension of a computation graph.  It is the
attention backend of the exact evaluators (no transformer weights): keys,
values, and queries are IEEE doubles computed straight from the graph's
expressions, and the winning key is found by hull search instead of a
softmax over all tokens.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

hull_ext = None


def load_hull_ext():
    """Compile (once) and return the pybind11 hull extension, or None."""
    global hull_ext
    if hull_ext is not None:
        return hull_ext
    try:
        from torch.utils.cpp_extension import load

        root = os.path.dirname(os.path.abspath(__file__))
        hull_ext = load(
            name="hull_ext",
            sources=[os.path.join(root, "hull_ext.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_include_paths=[root],
            verbose=False,
        )
        return hull_ext
    except Exception as e:
        logger.warning("Could not load hull extension: %s", e)
        return None


class HullAttention:
    """Per-lookup O(log n) attention using the pybind11 convex hull."""

    def __init__(self, lookup, ext):
        self.lookup = lookup
        nv = len(lookup.value_exprs)
        self.num_value_pairs = (nv + 1) // 2
        self.nv = nv
        self.cache = ext.HullKVCache(1, self.num_value_pairs)
        # Exact score ties (e.g. equidistant address keys whose inv_log_pos
        # tie-break term vanishes in double rounding) must resolve according
        # to the lookup's tie_break mode, not the C++ default of AVERAGE.
        latest = lookup.tie_break == "latest"
        for head in range(self.num_value_pairs):
            self.cache.set_tiebreak(0, head, latest)
        self.seq = -1
        # Reusable insert buffers (the cache copies data on insert).
        self.keys_buf = np.zeros((self.num_value_pairs, 2))
        self.values_buf = np.zeros((self.num_value_pairs, 2))
        self.values_flat = self.values_buf.reshape(-1)

    def clear(self):
        self.cache.clear()
        self.seq = -1

    def insert_raw(self, kx, ky, raw_vals):
        """Insert a precomputed key and value list, reusing scratch buffers."""
        keys = self.keys_buf
        keys[:, 0] = kx
        keys[:, 1] = ky
        self.values_flat[: self.nv] = raw_vals
        self.seq += 1
        self.cache.insert(0, keys, self.values_buf, self.seq)
