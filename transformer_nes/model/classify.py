"""Attention-serving classification licenses for the generated engine.

The engine replaces hull attention with O(1) state wherever a lookup's
queries license it: memory lookups whose query is a positive scalar
multiple of [local_address, 1] are served by the exact nearest-written-
address structure, and constant-query LATEST lookups by running-argmax
registers (replace-on->= at insert time reproduces the positional LATEST
tie-break exactly).  The numeric decisions live HERE so the weights
engine adapts its representation (queries as in_proj matrix rows) to
linear ``(coeff, id)`` terms against one set of licenses.  Classification is a
speed decision only -- anything unmatched keeps a real hull.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

Term = tuple[float, int]


def const_query(terms: Iterable[Term], one_id: int) -> float | None:
    """The value of a linear form that is a lone multiple of ``one`` (the
    empty form is the constant 0), else None."""
    terms = [(float(c), i) for c, i in terms if c]
    if not terms:
        return 0.0
    if len(terms) == 1 and terms[0][1] == one_id:
        return terms[0][0]
    return None


def memory_query(
    qx_terms: Iterable[Term],
    qy_terms: Iterable[Term],
    *,
    one_id: int,
) -> list[tuple[int, int]] | None:
    """The integer local-address terms of a structure-eligible memory query.

    Eligible when qy is a lone positive constant s on ``one`` (the hardmax
    temperature -- a positive scale multiplies every score, so the argmax
    is unchanged) and qx/s has integer coefficients below 2**24.  Returns
    ``[(int_coeff, id), ...]`` or None.
    """
    qy = const_query(qy_terms, one_id)
    if qy is None or qy <= 0.0:
        return None
    out: list[tuple[int, int]] = []
    for c, i in qx_terms:
        r = float(c) / qy
        if (
            not math.isfinite(r)
            or abs(r - round(r)) > 1e-9
            or abs(r) >= 2**24
        ):
            return None
        out.append((int(round(r)), i))
    return out


def register_query(
    qx_terms: Iterable[Term],
    qy_terms: Iterable[Term],
    *,
    one_id: int,
    latest: bool,
) -> tuple[float, float] | None:
    """The constant ``(qx, qy)`` of a running-argmax-eligible query.

    Needs the LATEST tie-break: with a constant query every token's score
    is a fixed function of its key, and replace-on->= in insertion order
    IS the positional LATEST rule.  Returns None otherwise.
    """
    if not latest:
        return None
    qx = const_query(qx_terms, one_id)
    qy = const_query(qy_terms, one_id)
    if qx is None or qy is None:
        return None
    return qx, qy


# clear_key gating shifts a key by -1e30; anything past this threshold is a
# gate term, and the live/gated test on the double key is "ky > -1e29".
CLEAR_GATE_MIN = 1e29


def integer_score_terms(
    qx: float,
    qy: float,
    kx_terms: Iterable[Term],
    ky_terms: Iterable[Term],
) -> tuple[list[Term], list[int]] | None:
    """The integer form of a running-argmax score, if licensed.

    With a constant query, qx == qy > 0 makes the score a positive multiple
    of kx + ky, so the integer sum preserves the argmax and its >= ties
    exactly.  ky's clear-gate terms (coeff <= -CLEAR_GATE_MIN on 0/1 gate
    dims) split out as an exact zero test replacing the "ky > -1e29" guard;
    after merging, every remaining coefficient must be an integer or +-0.5
    below 2**24.  Returns ``(score_terms, gate_ids)`` or None.
    """
    if not (math.isfinite(qx) and qx == qy and qx > 0.0):
        return None
    gates: list[int] = []
    acc: dict[int, float] = {}
    for terms, allow_gate in ((kx_terms, False), (ky_terms, True)):
        for c, i in terms:
            c = float(c)
            if not c:
                continue
            if allow_gate and c <= -CLEAR_GATE_MIN:
                gates.append(i)
                continue
            acc[i] = acc.get(i, 0.0) + c
    for c in acc.values():
        if (
            not math.isfinite(c)
            or abs(c) >= 2**24
            or (c != int(c) and abs(c) != 0.5)
        ):
            return None
    return [(c, i) for i, c in acc.items() if c], gates
