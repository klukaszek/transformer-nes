"""Shared attention-serving classification for the generated engines.

Both engines replace hull attention with O(1) state wherever a lookup's
queries license it: memory lookups whose query is a positive scalar
multiple of [local_address, 1] are served by the exact nearest-written-
address structure, and constant-query LATEST lookups by running-argmax
registers (replace-on->= at insert time reproduces the positional LATEST
tie-break exactly).  The numeric decisions live HERE so the weights
engine (queries as in_proj matrix rows) and the C++ kernel (queries as
compiled graph expressions) cannot drift apart; each adapts its
representation to linear ``(coeff, id)`` terms.  Classification is a
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
    banned_ids: Iterable[int] = (),
) -> list[tuple[int, int]] | None:
    """The integer local-address terms of a structure-eligible memory query.

    Eligible when qy is a lone positive constant s on ``one`` (the hardmax
    temperature -- a positive scale multiplies every score, so the argmax
    is unchanged) and qx/s has integer coefficients below 2**24 reading
    none of ``banned_ids`` (the position features).  Returns
    ``[(int_coeff, id), ...]`` or None.
    """
    qy = const_query(qy_terms, one_id)
    if qy is None or qy <= 0.0:
        return None
    banned_ids = set(banned_ids)
    out: list[tuple[int, int]] = []
    for c, i in qx_terms:
        r = float(c) / qy
        if (
            not math.isfinite(r)
            or i in banned_ids
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
