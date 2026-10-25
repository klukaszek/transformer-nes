"""Build-time CPU proofs: constants, conservative ranges and symbolic PCs.

These analyses reject execution that could cross a device boundary or hide
interior control flow. They inspect folded graphs and never execute a CPU.
"""

from __future__ import annotations

import numpy as np

from transformer_nes.cpu6502.spec import READS_MEMORY
from transformer_nes.graph.core import (
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
)
from transformer_nes.graph.evaluate import exact_bounds


def const_root(fold_result, root: str, k: int, label: str) -> int:
    one = fold_result.program_graph.one
    d = fold_result.dim_of_name[f"{root}__s{k}"]
    expr = getattr(d, "expr", None)
    if expr is None or any(t is not one for t in expr.terms):
        raise RuntimeError(f"{label} step {k}: {root} did not fold to a constant")
    return int(round(expr.terms.get(one, 0.0)))


def interval_of(dim) -> tuple[float, float]:
    """Conservative [lo, hi] bound of a folded dim over byte-valued inputs
    (bits give [0, 1], 'one' gives 1). Input-valued lookups inherit
    their sources' ranges; unsupported reads fail instead of guessing."""
    BIT_INPUTS = frozenset(
        {
            "is_state",
            "in_nmi",
            "in_irq",
            "in_halted",
            "in_p_c",
            "in_p_z",
            "in_p_i",
            "in_p_d",
            "in_p_b",
            "in_p_u",
            "in_p_v",
            "in_p_n",
        }
    )
    memo: dict = {}

    def ev(d):
        r = memo.get(d)
        if r is not None:
            return r
        if isinstance(d, InputDimension):
            if d.name == "one":
                r = (1.0, 1.0)
            elif d.name in BIT_INPUTS:
                r = (0.0, 1.0)
            else:
                r = (0.0, 255.0)
        elif isinstance(d, ReGLUDimension):
            (alo, ahi), bhi = evx(d.a_expr), evx(d.b_expr)[1]
            if bhi <= 0.0:
                r = (0.0, 0.0)
            else:
                # b in (0, bhi]: extremes at b -> 0 and b == bhi.
                r = (min(0.0, alo * bhi, ahi * bhi), max(0.0, alo * bhi, ahi * bhi))
        elif isinstance(d, PersistDimension):
            # These named fetch results come from byte-valued RAM even when
            # their exact masked-read circuit has a very loose naive
            # interval. Preserve that semantic range for symbolic shapes.
            r = (0.0, 255.0) if (d.name or "").startswith("fetched_operand_") else evx(d.expr)
        elif isinstance(d, LookUpDimension):
            # A lookup returns one of its insert-side values (registers
            # come in via state attention, so this is the live data
            # path), or 0 when nothing matched.
            lo, hi = evx(d.lookup.value_exprs[d.value_index])
            r = (min(lo, 0.0), max(hi, 0.0))
        else:
            raise RuntimeError(f"unbounded dim {d.name!r}")
        memo[d] = r
        return r

    def evx(e):
        lo = hi = 0.0
        for d, c in e.terms.items():
            dlo, dhi = ev(d)
            lo += min(c * dlo, c * dhi)
            hi += max(c * dlo, c * dhi)
        return (lo, hi)

    result = ev(dim)
    if result[0] < 0 or result[1] >= 0x2000:
        exact = exact_bounds(
            dim,
            lambda d: (1, 1) if d.name == "one" else (0, 1 if d.name in BIT_INPUTS else 255),
        )
        if exact is not None:
            return exact
    return result


def eval_over_heads(fold_result, dim, length):
    """Evaluate a folded dim as a function of the symbolic head pc, for every
    head where the window fits without wrapping $FFFF.  The head enters the
    graph through the step-0 pc persists (state arrives via attention, so
    raw inputs never carry it); evaluation cuts there and raises if the
    value depends on anything else -- the straight-line license needs
    next_pc to be a pure function of pc."""
    heads = np.arange(0x10000 - length + 1, dtype=np.float64)
    if "current_pc_lo__s0" in fold_result.dim_of_name:
        memo: dict = {
            fold_result.dim_of_name["current_pc_lo__s0"]: heads % 256.0,
            fold_result.dim_of_name["current_pc_hi__s0"]: heads // 256.0,
        }
    else:
        # Linear-inlined shape folds may substitute the byte persists into
        # the already-combined address while retaining the same symbolic
        # head license.
        memo = {fold_result.dim_of_name["current_pc__s0"]: heads}

    def ev(d):
        r = memo.get(d)
        if r is not None:
            return r
        if isinstance(d, InputDimension):
            if d.name != "one":
                raise RuntimeError(f"depends on input {d.name!r}")
            r = np.float64(1.0)
        elif isinstance(d, ReGLUDimension):
            a, b = np.asarray(evx(d.a_expr)), np.asarray(evx(d.b_expr))
            r = np.where(b > 0.0, a * b, 0.0)
        elif isinstance(d, PersistDimension):
            r = evx(d.expr)
        elif isinstance(d, LookUpDimension):
            raise RuntimeError("depends on a memory lookup")
        else:
            raise RuntimeError(f"unsupported dim {d.name!r}")
        memo[d] = r
        return r

    def evx(e):
        acc = np.float64(0.0)
        for d, c in e.terms.items():
            acc = acc + ev(d) * c
        return acc

    return heads, np.asarray(ev(dim))


def check_reads_licensed(fold_result, ops, label: str) -> None:
    """Only the first read crosses the request boundary; later reads must avoid devices.

    Fixed shapes contain no stores, so reads cannot depend on uncommitted
    writes from an earlier instruction in the same forward.
    """
    for k, op in enumerate(ops):
        if k and op.operation in READS_MEMORY and op.mode not in ("imp", "acc", "imm"):
            lo, hi = interval_of(fold_result.dim_of_name[f"ea__s{k}"])
            if not (0 <= lo <= hi < 0x2000 or 0x8000 <= lo <= hi < 0x10000):
                raise RuntimeError(f"{label}: step {k} read may reach a watched device")
