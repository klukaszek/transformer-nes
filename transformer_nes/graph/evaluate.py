"""Partial evaluation of a ProgramGraph under one pin assignment.

Evaluating a graph under pins substitutes, dim by dim in construction
order, what is known so far: a pinned persist is its constant, a ReGLU
whose gate folds to a non-positive constant is zero, one whose gate folds
positive is its linear operand scaled, and what cannot be decided stays a
dim of its own ("folded").  Every value is a linear combination of the
dims that stay, so it is a sparse row, and a dim's value is its
expression's row times the rows of the dims it reads.  The dims fall into
levels by dependency depth. Each level's expressions are cached as a sparse
matrix over preceding levels, so substitution is one sparse product followed
by elementwise folding rules. Lazy views decode only the expressions a
folder actually consumes.
"""

from __future__ import annotations

import math
import weakref

import numpy as np
from scipy import sparse

from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    MemoryReadDimension,
    PersistDimension,
    ProgramGraph,
    ReGLUDimension,
)

ABSENT, VALUE, FOLDED = 0, 1, 2  # a dim under a pin set: left as itself, a value, or folded
INDEX = np.int32  # scipy's own index type: arrays of it pass straight through


def enumerate_values(expr, domains, *, max_combinations=256):
    """Prove a small circuit's integer outputs by exhaustive F=1 evaluation.

    ``domains`` supplies inclusive integer ranges for input dimensions and
    memory-byte values, or None for unsupported leaves. Bits from one read
    share its byte's domain; different reads remain independent. Addresses
    and lookups are unsupported. No graph nodes or persistent caches are made.
    Refuse unsupported coefficients, overflow and values finer than half units.
    Returns (outputs, leaf values), or None when the proof cannot be completed.
    """
    leaves, seen, order = {}, set(), []
    limit = np.iinfo(np.int32).max

    def visit(d):
        if d in seen:
            return True
        seen.add(d)
        if isinstance(d, (InputDimension, MemoryReadDimension)):
            if isinstance(d, MemoryReadDimension):
                if d.output_kind == "address":
                    return False
                leaf = d.memory_read.dims[0]
            else:
                leaf = d
            domain = domains(leaf)
            if domain is None:
                return False
            lo, hi = domain
            if (not all(math.isfinite(v) and v == int(v) for v in domain)
                    or lo > hi or max(abs(lo), abs(hi)) > limit // 2):
                return False
            leaves[leaf] = range(int(lo), int(hi) + 1)
            if math.prod(map(len, leaves.values())) > max_combinations:
                return False
        elif isinstance(d, (PersistDimension, ReGLUDimension)):
            expressions = (d.expr,) if isinstance(d, PersistDimension) else (d.a_expr, d.b_expr)
            if not all(visit(source) for e in expressions for source in e.terms):
                return False
        else:
            return False
        order.append(d)
        return True

    if not all(visit(d) for d in expr.terms):
        return None
    grids = np.meshgrid(*leaves.values(), indexing="ij")
    inputs = {d: grid.reshape(-1) for d, grid in zip(leaves, grids, strict=True)}
    values = {d: 2 * v for d, v in inputs.items()}

    def expression(e):
        result = 0
        for d, c in e.terms.items():
            if not math.isfinite(c) or 2 * c != int(2 * c) or abs(2 * c) > limit:
                return None
            # Encoded int32 operands fit int64 products. Each individual
            # half coefficient and ReGLU product must remain on the F=1 grid.
            term = values[d] * int(2 * c)
            if np.any(term % 2):
                return None
            term = term // 2
            if np.max(np.abs(term)) > limit:
                return None
            result = result + term
            if np.max(np.abs(result)) > limit:
                return None
        return result

    for d in order:
        if isinstance(d, MemoryReadDimension) and d.output_kind.startswith("bit"):
            byte = inputs[d.memory_read.dims[0]]
            values[d] = 2 * ((byte >> int(d.output_kind[3:])) & 1)
        elif isinstance(d, PersistDimension):
            values[d] = expression(d.expr)
        elif isinstance(d, ReGLUDimension):
            a, b = expression(d.a_expr), expression(d.b_expr)
            if a is None or b is None:
                return None
            product = a * np.maximum(0, b)
            if np.any(product % 2):
                return None
            values[d] = product // 2
        if values[d] is None or np.max(np.abs(values[d])) > limit:
            return None
    result = expression(expr)
    if result is None or np.any(result % 2):
        return None
    return np.asarray(result) // 2, inputs


def exact_bounds(dim, input_bounds) -> tuple[float, float] | None:
    """Exhaustive bounds over at most 256 input combinations, else None.

    Memory/lookup dependencies stay unknown. Evaluation uses exact half-unit
    arithmetic with int32 bounds and requires integer outputs.
    """
    proof = enumerate_values(
        dim.as_expr(), lambda d: input_bounds(d) if isinstance(d, InputDimension) else None,
    )
    if proof is None:
        return None
    values, _inputs = proof
    return float(np.min(values)), float(np.max(values))


EVALUATORS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # graph -> {structural_reads: evaluator}


def evaluator(pg: ProgramGraph, structural_reads: bool) -> GraphEvaluator:
    """`pg`'s evaluator, built once."""
    per_graph = EVALUATORS.setdefault(pg, {})
    if structural_reads not in per_graph:
        per_graph[structural_reads] = GraphEvaluator(pg, structural_reads)
    return per_graph[structural_reads]


class GraphEvaluator:
    """`pg`'s dims by level, each level's expressions as sparse rows: a and
    b per ReGLU; per persist its expression and its multiply_of factors
    (with structural reads; empty rows where it has none).  A dim's
    position orders the dims level by level."""

    def __init__(self, pg: ProgramGraph, structural_reads: bool):
        dims = list(pg.all_dims)
        if pg.one not in dims:
            dims.insert(0, pg.one)
        self.dims, self.one = dims, pg.one
        self.column = {d: i for i, d in enumerate(dims)}
        self.one_column = self.column[pg.one]
        lookup_persists = {
            d for lu in pg.all_lookups
            # Structural reads accept a linear address directly. Keeping
            # query-only intermediates opaque needlessly adds layers after
            # the original attention fabric has been replaced by a read.
            # Keys and values stay opaque: clear gates and shared insertion
            # state still depend on their exact structure.
            for e in (lu.key_exprs + lu.value_exprs
                      + ([] if structural_reads else lu.query_exprs))
            for d in e.terms if isinstance(d, PersistDimension)
        }

        def site(d):
            return getattr(d, "memory_read", None)

        def products(d):
            return (getattr(d, "multiply_of", None) if structural_reads else None) or (Expression(),) * 2

        def reads(d) -> list[Expression]:
            if isinstance(d, ReGLUDimension):
                return [d.a_expr, d.b_expr]
            if isinstance(d, PersistDimension):
                return [d.expr, *products(d)]
            return []

        depth: dict = {}
        for d in dims:
            evaluated = isinstance(d, (ReGLUDimension, PersistDimension))
            depth[d] = (1 + max((depth[t] for e in reads(d) for t in e.terms), default=0)
                        if evaluated else 0)
        levels: list[list] = [[] for _ in range(max(depth.values()) + 1)]
        for d in dims:
            levels[depth[d]].append(d)
        self.n_regs = [0]
        for k in range(1, len(levels)):  # a level: its ReGLUs, then its persists
            regs = [d for d in levels[k] if isinstance(d, ReGLUDimension)]
            levels[k] = regs + [d for d in levels[k] if isinstance(d, PersistDimension)]
            self.n_regs.append(len(regs))
        self.levels = levels
        ordered = [d for level in levels for d in level]
        self.position = {d: p for p, d in enumerate(ordered)}
        self.level_of = {d: (k, i) for k, level in enumerate(levels) for i, d in enumerate(level)}
        self.level_start = np.cumsum([0] + [len(level) for level in levels])
        self.level_columns = [np.array([self.column[d] for d in level], INDEX) for level in levels]

        self.stacks = [None]
        for k, level in enumerate(levels[1:], 1):
            regs, persists = level[:self.n_regs[k]], level[self.n_regs[k]:]
            self.stacks.append(self.stack(
                [d.a_expr for d in regs] + [d.b_expr for d in regs]
                + [d.expr for d in persists]
                + [products(d)[0] for d in persists] + [products(d)[1] for d in persists],
                int(self.level_start[k])))
        self.lookups = list(pg.all_lookups)
        self.lookup_rows, row = {}, 0  # lookup -> (its first row, its query/key/value counts)
        for lu in self.lookups:
            counts = (len(lu.query_exprs), len(lu.key_exprs), len(lu.value_exprs))
            self.lookup_rows[lu] = (row, counts)
            row += sum(counts)
        self.lookup_stack = self.stack(
            [e for lu in self.lookups for e in lu.query_exprs + lu.key_exprs + lu.value_exprs],
            len(ordered))

        self.by_name: dict[str, list[int]] = {}
        for d in ordered:
            if isinstance(d, PersistDimension):
                self.by_name.setdefault(d.name, []).append(self.position[d])
        self.inlinable = np.array([
            isinstance(d, PersistDimension) and d not in lookup_persists and site(d) is None
            and not (structural_reads and getattr(d, "read_bit", None) is not None)
            for d in ordered])
        self.has_product = np.array([structural_reads and getattr(d, "multiply_of", None) is not None
                                     for d in ordered])
        self.hazards: dict = {}  # GraphFolder.insert_hazards, by pin names

    def stack(self, exprs, width: int) -> sparse.csr_matrix:
        """Expressions over dimensions in preceding levels, preserving term order."""
        indptr, indices, coefficients = [0], [], []
        for expr in exprs:
            indices.extend(self.position[d] for d in expr.terms)
            coefficients.extend(expr.terms.values())
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(coefficients, np.float64), np.asarray(indices, INDEX),
             np.asarray(indptr, INDEX)), shape=(len(exprs), width),
        )

    def evaluate(self, pins: dict, frozen=frozenset(), *, inline_linear: bool = False,
                 preserve=frozenset()) -> Evaluation:
        n = len(self.position)
        pinned = np.zeros(n, bool)
        pin_value = np.zeros(n)
        frozen_dims = np.zeros(n, bool)
        preserved = np.zeros(n, bool)
        for name, value in pins.items():
            for position in self.by_name.get(name, ()):
                pinned[position], pin_value[position] = True, value
        for name in frozen:
            for position in self.by_name.get(name, ()):
                frozen_dims[position] = True
        for dim in preserve:
            preserved[self.position[dim]] = True

        status = np.zeros(n, np.int8)
        status[self.position[self.one]] = VALUE
        level0 = self.level_columns[0]
        values = Rows(np.arange(len(level0) + 1), level0, np.ones(len(level0)), len(self.dims))
        parts = [None]
        for k in range(1, len(self.levels)):
            level, start, n_regs = self.levels[k], self.level_start[k], self.n_regs[k]
            n_persists = len(level) - n_regs
            stacked = self.stacks[k] @ values.matrix()
            stacked.eliminate_zeros()
            stacked.sort_indices()
            is_const, value = constant_rows(stacked, self.one_column)

            def block(i, m=n_persists):
                return slice(i, i + m)

            a, b_ = block(0, n_regs), block(n_regs, n_regs)
            e, ma, mb = (block(2 * n_regs + i * n_persists) for i in range(3))
            regs, persists = slice(0, n_regs), slice(n_regs, len(level))
            source = np.full(len(level), -1)  # the stacked row each row takes, scaled
            scale = np.zeros(len(level))
            const = np.zeros(len(level))
            own = np.zeros(len(level), bool)  # the dim stays as itself

            # ReGLUs: relu(b) * a.
            zero = (is_const[b_] & (value[b_] <= 0)) | (is_const[a] & (value[a] == 0))
            both = ~zero & is_const[a] & is_const[b_]
            linear = ~zero & ~both & is_const[b_]
            own[regs] = ~(zero | both | linear)
            const[regs] = np.where(both, value[a] * value[b_], 0.0)
            source[regs] = np.where(linear, np.arange(a.start, a.stop), -1)
            scale[regs] = value[b_]

            # Persists: frozen, pinned, a product, or their expression.
            dims = slice(start + n_regs, start + len(level))
            p_frozen = frozen_dims[dims]
            open_ = ~p_frozen & ~pinned[dims]
            p_const = np.where(open_, 0.0, pin_value[dims])
            product = self.has_product[dims] & open_
            by_ma = product & is_const[ma]
            by_mb = product & ~is_const[ma] & is_const[mb]
            open_ &= ~(by_ma | by_mb)
            inlined = (open_ & ~is_const[e] & self.inlinable[dims]
                       & ~preserved[dims] & inline_linear)
            own[persists] = (open_ & ~is_const[e] & ~inlined) | p_frozen
            const[persists] = np.where(open_ & is_const[e], value[e], p_const)
            source[persists] = np.select(
                [by_ma, by_mb, inlined],
                [np.arange(mb.start, mb.stop), np.arange(ma.start, ma.stop), np.arange(e.start, e.stop)], -1)
            scale[persists] = np.select([by_ma, by_mb], [value[ma], value[mb]], 1.0)

            values.append(*level_rows(stacked, source, scale, own, const,
                                      self.level_columns[k], self.one_column))
            left = np.concatenate([np.zeros(n_regs, bool), p_frozen])
            status[start:start + len(level)] = np.where(
                own, np.where(left, ABSENT, FOLDED), VALUE)
            parts.append((stacked, a, b_, e, n_regs))
        everything = values.matrix()
        lookups = self.lookup_stack @ everything
        lookups.eliminate_zeros()
        lookups.sort_indices()
        return Evaluation(self, everything, status, parts, lookups)


def constant_rows(m: sparse.csr_matrix, one: int) -> tuple[np.ndarray, np.ndarray]:
    """Which rows of `m` are constants (terms on `one` alone), and their values."""
    per_row = np.diff(m.indptr)
    value = np.zeros(m.shape[0])
    on_one = m.indices == one
    value[np.repeat(np.arange(m.shape[0]), per_row)[on_one]] = m.data[on_one]
    return per_row == (value != 0), value


def level_rows(stacked, source, scale, own, const, columns, one):
    """A level's rows as CSR arrays: row i is stacked row source[i] times
    scale[i], or (own) the dim at columns[i], or the constant const[i]."""
    taken = np.flatnonzero(source >= 0)
    counts = np.diff(stacked.indptr)[source[taken]]
    at = (np.repeat(stacked.indptr[source[taken]] - (np.cumsum(counts) - counts), counts)
          + np.arange(counts.sum()))
    kept, constant = np.flatnonzero(own), np.flatnonzero(~own & (const != 0))
    rows = np.concatenate([np.repeat(taken, counts), kept, constant])
    cols = np.concatenate([stacked.indices[at], columns[kept], np.full(len(constant), one)])
    data = np.concatenate([stacked.data[at] * np.repeat(scale[taken], counts),
                           np.ones(len(kept)), const[constant]])
    nonzero = data != 0
    rows, cols, data = rows[nonzero], cols[nonzero], data[nonzero]
    order = np.argsort(rows, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(source)))])
    return indptr, cols[order], data[order]


class Rows:
    """CSR rows grown by blocks, readable at any point without copying
    what came before (the arrays grow by doubling)."""

    def __init__(self, indptr, indices, data, width: int):
        self.width = width
        self.indptr = np.asarray(indptr, INDEX)
        self.indices, self.data = np.asarray(indices, INDEX), np.asarray(data, np.float64)
        self.nnz = len(self.data)

    def append(self, indptr, indices, data) -> None:
        need = self.nnz + len(data)
        if need > len(self.data):
            grow = max(need, 2 * len(self.data))
            self.data, self.indices = np.resize(self.data, grow), np.resize(self.indices, grow)
        self.data[self.nnz:need], self.indices[self.nnz:need] = data, indices
        self.indptr = np.concatenate([self.indptr, (indptr[1:] + self.nnz).astype(INDEX)])
        self.nnz = need

    def matrix(self) -> sparse.csr_matrix:
        return sparse.csr_matrix((self.data[:self.nnz], self.indices[:self.nnz], self.indptr),
                                 shape=(len(self.indptr) - 1, self.width))


class Evaluation:
    """One pin set's lazy views: substituted values, surviving operations,
    and lookup expressions. Sparse rows are decoded only when consumed."""

    def __init__(self, evaluator, values, status, parts, lookups):
        self.evaluator, self.values, self.parts, self.lookups = evaluator, values, parts, lookups
        self.statuses = status.tobytes()
        self.val, self.folded, self.lookup_exprs = Values(self), Folded(self), LookupExpressions(self)

    def expression(self, matrix: sparse.csr_matrix, row: int) -> Expression:
        lo, hi = matrix.indptr[row], matrix.indptr[row + 1]
        dims = self.evaluator.dims
        return Expression({dims[c]: float(v) for c, v in zip(matrix.indices[lo:hi], matrix.data[lo:hi], strict=True)})

    def status(self, dim) -> int:
        position = self.evaluator.position.get(dim)
        return ABSENT if position is None else self.statuses[position]

    def __iter__(self):
        return iter((self.val, self.folded, self.lookup_exprs))


class Values:
    def __init__(self, evaluation: Evaluation):
        self.evaluation, self.cache = evaluation, {}

    def __contains__(self, d) -> bool:
        return self.evaluation.status(d) == VALUE

    def __getitem__(self, d) -> Expression:
        if d not in self.cache:
            if d not in self:
                raise KeyError(d)
            evaluation = self.evaluation
            row = evaluation.evaluator.position[d]
            self.cache[d] = evaluation.expression(evaluation.values, row)
        return self.cache[d]

    def get(self, d, default=None):
        return self[d] if d in self else default  # noqa: SIM401 (this is get)

    def items(self):
        for d in self.evaluation.evaluator.dims:
            if d in self:
                yield d, self[d]


class Folded:
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def __contains__(self, d) -> bool:
        return self.evaluation.status(d) == FOLDED

    def __getitem__(self, d) -> tuple[Expression, ...]:
        if d not in self:
            raise KeyError(d)
        evaluation = self.evaluation
        k, i = evaluation.evaluator.level_of[d]
        stacked, a, b, e, n_regs = evaluation.parts[k]
        if i < n_regs:
            return (evaluation.expression(stacked, a.start + i),
                    evaluation.expression(stacked, b.start + i))
        return (evaluation.expression(stacked, e.start + i - n_regs),)


class LookupExpressions:
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation
        self.first = evaluation.evaluator.lookup_rows

    def __contains__(self, lu) -> bool:
        return lu in self.first

    def __getitem__(self, lu) -> dict:
        evaluation = self.evaluation
        start, (nq, nk, nv) = self.first[lu]

        def exprs(at, count):
            return [evaluation.expression(evaluation.lookups, at + j) for j in range(count)]

        return {"query": exprs(start, nq), "key": exprs(start + nq, nk), "values": exprs(start + nq + nk, nv)}

    def get(self, lu, default=None):
        return self[lu] if lu in self else default  # noqa: SIM401 (this is get)
