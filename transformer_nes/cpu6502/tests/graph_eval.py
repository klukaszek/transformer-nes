"""Test-only oracle for fold.py: runs a ProgramGraph directly, in float64.

Executes the graph itself -- one token per call, dict arithmetic, hardmax
attention -- so a folded graph can be checked against something that shares
none of the engine's schedule/weights/codegen machinery.  O(tokens) per query.
"""

from __future__ import annotations

import math

from transformer_nes.graph.core import (
    Dimension,
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ProgramGraph,
    ReGLUDimension,
)


class ProgramEvaluator:
    """Evaluate a ProgramGraph token by token in float64."""

    def __init__(self, program_graph: ProgramGraph):
        self.pg = program_graph
        for lu in program_graph.all_lookups:
            if lu.tie_break != "latest":
                raise NotImplementedError(
                    f"lookup {lu.name!r}: only the LATEST tie-break is supported"
                )
        self.logs: dict[int, list] = {lu.id: [] for lu in program_graph.all_lookups}
        self.dim_of_name: dict[str, Dimension] = {}
        for d in program_graph.all_dims:
            self.dim_of_name.setdefault(d.name, d)
        self.pos = 0

    def step(self, inputs: dict[str, float]) -> dict[Dimension, float]:
        """One token: set inputs by dim name, evaluate every dim in creation
        (= topological) order, insert-then-query every lookup.  Returns the
        full value map for this position."""
        pg = self.pg
        self.pos += 1
        v: dict[Dimension, float] = {pg.one: 1.0}
        for d in pg.all_dims:
            if isinstance(d, InputDimension):
                if d not in v:
                    v[d] = float(inputs.get(d.name, 0.0))
            elif isinstance(d, ReGLUDimension):
                b = d.b_expr.evaluate(v)
                v[d] = d.a_expr.evaluate(v) * b if b > 0.0 else 0.0
            elif isinstance(d, PersistDimension):
                v[d] = d.expr.evaluate(v)
            elif isinstance(d, LookUpDimension) and d.value_index == 0:
                self.attend(d.lookup, v)
        return v

    def attend(self, lu, v: dict) -> None:
        kx = lu.key_exprs_2d[0].evaluate(v)
        ky = lu.key_exprs_2d[1].evaluate(v)
        vals = [e.evaluate(v) for e in lu.value_exprs]
        log = self.logs[lu.id]
        log.append((kx, ky, vals))
        qx = lu.query_exprs_2d[0].evaluate(v)
        qy = lu.query_exprs_2d[1].evaluate(v)
        best = None
        best_score = -math.inf
        for tkx, tky, tvals in log:  # >=: exact ties go to the later token
            score = qx * tkx + qy * tky
            if score >= best_score:
                best_score = score
                best = tvals
        assert best is not None  # a non-empty log always has an argmax
        for i, dim in enumerate(lu.dims):
            v[dim] = best[i]

    def read(self, values: dict, name: str) -> float:
        return values.get(self.dim_of_name[name], 0.0)
