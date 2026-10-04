"""Exact matrix serialization shared by compiled models and checkpoints."""

import numpy as np
import scipy.sparse as sp


def csr_tensors(name: str, mat: np.ndarray, out: dict) -> None:
    m = sp.csr_matrix(mat)
    m.sort_indices()
    out[f"{name}.data"] = to_fixed_point(m.data, f"{name}.data")
    out[f"{name}.indices"] = m.indices.astype(np.int32)
    out[f"{name}.indptr"] = m.indptr.astype(np.int32)


def prune_layer(ip, op, fi, fo, num_heads, d_ffn):
    """Drop dead heads and FFN units from one layer's matrices, exactly.

    A head contributes 0 to the residual when its V rows or its out_proj
    columns are all zero (zero Q/K with live V is a passthrough -- kept);
    an FFN unit contributes 0 when its gate row, value row, or ff_out
    column is all zero.  Returns ragged matrices plus the kept counts.
    """
    k_base, v_base = 2 * num_heads, 4 * num_heads
    heads = [
        h
        for h in range(num_heads)
        if ip[v_base + 2 * h : v_base + 2 * h + 2].any() and op[:, 2 * h : 2 * h + 2].any()
    ]
    hi = [r for h in heads for r in (2 * h, 2 * h + 1)]
    ip = np.vstack([ip[hi], ip[[k_base + r for r in hi]], ip[[v_base + r for r in hi]]])
    op = op[:, hi]
    units = [j for j in range(d_ffn) if fi[j].any() and fi[d_ffn + j].any() and fo[:, j].any()]
    fi = np.vstack([fi[units], fi[[d_ffn + j for j in units]]])
    fo = fo[:, units]
    return ip, op, fi, fo, heads, units


def rescale_head_queries(in_proj: np.ndarray, num_heads: int) -> np.ndarray:
    """Divide each head's query rows by their construction temperature.

    The construction builds every head as softmax at a vast temperature
    (C = 1e10*sqrt(2) on memory queries), its argmax limit; dividing it
    out leaves integer scores, which the exported attention_temperature
    then scales just enough for softmax to be exact.  Only applied when
    the quotients land on integers (guarded), so the exported coefficients
    are integers -- the clear gates stay K-side and are never scaled.
    """
    out = in_proj.copy()
    for h in range(num_heads):
        rows = out[2 * h : 2 * h + 2]
        nz = np.abs(rows[rows != 0.0])
        if nz.size == 0:
            continue
        s = nz.min()
        if s <= 1.0:
            continue
        scaled = rows / s
        ints = np.rint(scaled)
        if (np.abs(scaled - ints) <= 1e-6 * np.maximum(1.0, np.abs(ints))).all():
            out[2 * h : 2 * h + 2] = ints
    return out


CLEAR_GATE_THRESHOLD = -1e29
# Softmax over integer scores whose best leads by >= 1: at this temperature
# the other weights (< e**-32 each) cannot move a rounded head output.
ATTENTION_TEMPERATURE = 32.0
SCALE_BITS = 1


def split_clear_gates(ip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift the -1e30 clear gates out of a layer's Q/K/V projection.

    A clear gate exists only to lose the argmax, never to be exact, so it is
    a *predicate*, not a coefficient: token t is masked for head h when the
    gated input is non-zero.  Extracting it leaves every surviving
    coefficient finite and integral, which is what lets the artifact be
    int32 with no magnitude reasoning anywhere.
    """
    mask = ip <= CLEAR_GATE_THRESHOLD
    kept = np.where(mask, 0.0, ip)
    rows, cols = np.nonzero(mask)
    return (
        kept,
        np.ascontiguousarray(rows, dtype=np.int32),
        np.ascontiguousarray(cols, dtype=np.int32),
    )


def to_fixed_point(arr: np.ndarray, name: str) -> np.ndarray:
    """Exact int32 in units of 2**-SCALE_BITS; raises if any value is lost."""
    scaled = arr * (1 << SCALE_BITS)
    ints = np.rint(scaled)
    if not np.array_equal(scaled, ints):
        bad = scaled[scaled != ints]
        raise AssertionError(
            f"{name}: {bad.size} coefficients are not exact at "
            f"SCALE_BITS={SCALE_BITS} (e.g. {bad[:3]})"
        )
    if np.abs(ints).max(initial=0.0) > np.iinfo(np.int32).max:
        raise AssertionError(f"{name}: coefficient exceeds int32")
    return ints.astype(np.int32)
