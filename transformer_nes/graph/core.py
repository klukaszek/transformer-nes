"""Computation-graph primitives that compile to transformer weights.

A program is a graph of dimensions -- named scalars of one token's
residual stream -- and every kind of dimension is something a transformer
layer computes exactly:

- InputDimension: a value the token embeds.
- ReGLUDimension: relu(b) * a, one feed-forward neuron.
- PersistDimension: a linear combination stored in its own slot.
- LookUpDimension: hardmax attention over every token so far, latest
  wins ties -- how a token reads earlier tokens (memory, state).

Expressions are linear combinations of dimensions; the helpers below
(reglu, stepglu, make_multiply, fetch) build the idioms everything else
uses (graph/ops.py has the arithmetic and memory ones).
model/schedule.py assigns dimensions to layers and model/weights.py turns
them into weights.  Folded experts (graph/fold.py) may also contain a
MemoryRead: one exact read the engine serves directly, in place of the
grouped lookups that compute it in the full graph.

Construction state is module-global (graph_dims, the caches); reset_graph
starts a new graph.
"""

from __future__ import annotations

from typing import TypeAlias

BIG = 1e30  # large constant used to effectively zero-out attention keys (clear_key mechanism)

graph_dims = []
graph_lookups = []
graph_memory_reads = []
multiply_cache = {}
reglu_cache = {}
stepglu_cache = {}
clear_key_cache = {}


def expr_key(expr):
    return tuple(sorted((id(d), c) for d, c in expr.terms.items()))


class Expression:
    """A linear combination of dimensions: {dimension: coefficient}, with
    the constant term as a coefficient on `one`.  Supports +, - and scalar
    *; zero coefficients are dropped."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict | None = None):
        if terms is None:
            self.terms = {}
        elif isinstance(terms, dict):
            self.terms = {k: v for k, v in terms.items() if v != 0}
        else:
            raise TypeError

    def copy(self) -> Expression:
        return Expression(dict(self.terms))

    def __add__(self, other) -> Expression:
        if isinstance(other, (int, float)):
            if other == 0:
                return self.copy()
            return self + Expression({one: other})
        if isinstance(other, Dimension):
            other = Expression({other: 1})
        if isinstance(other, Expression):
            r = dict(self.terms)
            for d, c in other.terms.items():
                r[d] = r.get(d, 0) + c
                if r[d] == 0:
                    del r[d]
            return Expression(r)
        return NotImplemented

    def __radd__(self, other) -> Expression:
        if isinstance(other, (int, float)):
            if other == 0:
                return self.copy()
            return Expression({one: other}) + self
        if isinstance(other, Dimension):
            return Expression({other: 1}) + self
        return NotImplemented

    def __sub__(self, other) -> Expression:
        if isinstance(other, (int, float)):
            return self + (-other)
        if isinstance(other, Dimension):
            return self + Expression({other: -1})
        if isinstance(other, Expression):
            r = dict(self.terms)
            for d, c in other.terms.items():
                r[d] = r.get(d, 0) - c
                if r[d] == 0:
                    del r[d]
            return Expression(r)
        return NotImplemented

    def __rsub__(self, other) -> Expression:
        neg = Expression({d: -c for d, c in self.terms.items()})
        if isinstance(other, (int, float)):
            return neg + other
        if isinstance(other, Dimension):
            return Expression({other: 1}) + neg
        if isinstance(other, Expression):
            return other + neg
        return NotImplemented

    def __mul__(self, other) -> Expression:
        if isinstance(other, (int, float)):
            if other == 0:
                return Expression()
            return Expression({d: c * other for d, c in self.terms.items()})
        return NotImplemented

    def __rmul__(self, other) -> Expression:
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> Expression:
        return Expression({d: -c for d, c in self.terms.items()})

    def __getitem__(self, dim):
        return self.terms.get(dim, 0)

    def __setitem__(self, dim, value):
        if value == 0 and dim in self.terms:
            del self.terms[dim]
        elif value != 0:
            self.terms[dim] = value

class Dimension:
    """One scalar of a token's residual stream.  Ids are allocated in
    construction order, which the scheduler and the engines' digests
    depend on: building the same graph twice gives the same weights."""

    counter = 0

    def __init__(self, name=None, kind="generic"):
        self.id = Dimension.counter
        Dimension.counter += 1
        self.name = name or f"dim_{self.id}"
        self.kind = kind
        graph_dims.append(self)

    def as_expr(self):
        return Expression({self: 1})

    def __add__(self, o):
        return self.as_expr().__add__(o)

    def __radd__(self, o):
        return self.as_expr().__radd__(o)

    def __sub__(self, o):
        return self.as_expr().__sub__(o)

    def __rsub__(self, o):
        return self.as_expr().__rsub__(o)

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            if o == 0:
                return Expression()
            return Expression({self: o})
        return NotImplemented

    def __rmul__(self, o):
        if isinstance(o, (int, float)):
            if o == 0:
                return Expression()
            return Expression({self: o})
        return NotImplemented

    def __neg__(self):
        return Expression({self: -1})

    def __repr__(self):
        return f"{self.kind}:{self.name}[{self.id}]"


class InputDimension(Dimension):
    """A value the token's embedding sets."""

    def __init__(self, name):
        super().__init__(name, kind="input")


one = InputDimension("one")  # the constant 1, set on every token


class PersistDimension(Dimension):
    """A dimension that stores a linear combination in a dedicated slot.

    Unlike ReGLU (which gates), persist is a pure linear projection realized
    through ff_out. It groups multiple dims into one slot, reducing d_model.
    Treated as a schedulable gate (own phase) for pathwidth/NL purposes.
    """

    def __init__(self, expr, name=None):
        super().__init__(name or f"persist_{Dimension.counter}", kind="persist")
        self.expr = expr


class ReGLUDimension(Dimension):
    """relu(b) * a: one feed-forward neuron of an FFN block."""

    def __init__(self, a_expr, b_expr, name=None):
        super().__init__(name or f"reglu_{Dimension.counter}", kind="reglu")
        self.a_expr = a_expr
        self.b_expr = b_expr


# memory_group sentinel for a fused 64K-span memory read (expert folds
# only): one full-address query, group computed at serve time.  The union
# keeps its 16 groups, whose purpose is the f32 attention-score range of
# the exported artifact.
MEMORY_GROUP_FUSED = -2


class LookUp:
    """One attention head: the 2-D query of this token against the 2-D key
    of every token so far (itself included); the best score wins, ties to
    the latest token, and its value expressions become this token's
    LookUpDimensions.  Keys are built so the intended token scores
    strictly highest (e.g. a parabola peaked at the queried address)."""

    counter = 0

    def __init__(self, value_exprs, query_exprs_2d, key_exprs_2d, tie_break="latest"):
        self.id = LookUp.counter
        LookUp.counter += 1
        self.name = None
        self.value_exprs = value_exprs
        self.query_exprs_2d = query_exprs_2d  # [qx_expr, qy_expr]
        self.key_exprs_2d = key_exprs_2d  # [kx_expr, ky_expr]
        self.tie_break = tie_break
        self.memory_group: int | None = None  # tagged by memory_at / folds
        self.state_fetch = False  # latest-STATE fetch: an identity for
        # structural backends, which embed the current state directly
        self.dims = [LookUpDimension(self, i) for i in range(len(value_exprs))]
        graph_lookups.append(self)


class LookUpDimension(Dimension):
    """The value_index-th value a LookUp returns."""

    def __init__(self, lookup, value_index):
        super().__init__(f"lookup_{lookup.id}_v{value_index}", kind="lookup")
        self.lookup = lookup
        self.value_index = value_index


class MemoryRead:
    """An exact memory read the engine serves directly (folded experts
    only): the latest written value at `address`, that address, and
    optionally the value's bits."""

    counter = 0

    def __init__(self, address, *, bits=()):
        self.id = MemoryRead.counter
        MemoryRead.counter += 1
        self.name = f"memory_read_{self.id}"
        self.address = to_expr(address)
        self.output_kinds = ("value", "address", *tuple(bits))
        self.dims = [
            MemoryReadDimension(self, i, kind)
            for i, kind in enumerate(self.output_kinds)
        ]
        graph_memory_reads.append(self)


class MemoryReadDimension(Dimension):
    """One output of a MemoryRead (value, address, or a bit)."""

    def __init__(self, memory_read, value_index, output_kind):
        super().__init__(
            f"memory_read_{memory_read.id}_{output_kind}", kind="memory_read"
        )
        self.memory_read = memory_read
        self.value_index = value_index
        self.output_kind = output_kind


def make_multiply(a, b):
    ka, kb = expr_key(a), expr_key(b)
    key = (ka, kb)
    if key in multiply_cache:
        return multiply_cache[key]
    neg_b = Expression({d: -c for d, c in b.terms.items()})
    r1 = ReGLUDimension(a, b)
    r2 = ReGLUDimension(a, neg_b)
    result = persist(Expression({r1: 1, r2: -1}))
    # Folds use this to collapse the product to a linear map when either
    # side pins to a constant (the ReGLU pair only has a rule for b).
    next(iter(result.terms)).multiply_of = (a, b)
    multiply_cache[key] = result
    return result


# What to_expr coerces: the DSL's helpers accept any of these.
ExprLike: TypeAlias = "Expression | Dimension | float"


def to_expr(x: Expression | Dimension | float) -> Expression:
    if isinstance(x, Expression):
        return x
    if isinstance(x, Dimension):
        return Expression({x: 1})
    if isinstance(x, (int, float)):
        if x == 0:
            return Expression()
        return Expression({one: x})
    raise TypeError(f"Cannot convert {type(x)} to Expression")


def reglu(a, b):
    """relu(b) * a — single ReGLU dimension.

    Use when b is known non-negative, giving reglu(a, b) = a * b.
    """
    a_expr = to_expr(a)
    b_expr = to_expr(b)
    key = (expr_key(a_expr), expr_key(b_expr))
    if key in reglu_cache:
        return Expression({reglu_cache[key]: 1})
    r = ReGLUDimension(a_expr, b_expr)
    reglu_cache[key] = r
    return Expression({r: 1})


def stepglu(a, b):
    """a * step(b >= 0) — two ReGLU dims combined via persist.

    Equals reglu(a, b + 1) - reglu(a, b).
    For integer b: equals a when b >= 0, 0 when b < 0.
    The two ReGLU dims are internal to the FFN; the persist dim
    stores the difference in a single residual slot.
    """
    a_expr = to_expr(a)
    b_expr = to_expr(b)
    key = (expr_key(a_expr), expr_key(b_expr))
    if key in stepglu_cache:
        return stepglu_cache[key]
    r1 = ReGLUDimension(a_expr, b_expr + Expression({one: 1}))
    r2 = ReGLUDimension(a_expr, b_expr)
    result = persist(Expression({r1: 1, r2: -1}))
    stepglu_cache[key] = result
    return result


def persist(expr, name=None):
    """Materialize a linear expression into a dedicated slot.

    Creates a PersistDimension that stores the value of expr. This reduces
    d_model by allowing the expression's constituents to die earlier.
    Treated as a schedulable operation (separate phase) for scheduling.
    """
    expr = to_expr(expr)
    dim = PersistDimension(expr, name=name)
    return Expression({dim: 1})


def to_2d_key(k, clear_key_expr=None, tie_break="latest"):
    """Map 1D key + optional clear_key to 2D key expressions via ReGLU.

    Scores carry no recency term: exact ties are the tie-break's job
    (TieBreak::LATEST, positional), which keeps every key coefficient an
    integer function of the key expression.
    """
    if tie_break != "latest":
        raise ValueError(f"only the LATEST tie-break is constructible, not {tie_break!r}")
    if len(k.terms) == 1 and one in k.terms:
        c = k.terms[one]
        k_abs = Expression({one: c * c})
    else:
        k_abs = make_multiply(k, k)
    kx = k * 2
    ky = -k_abs
    if clear_key_expr is not None:
        if len(clear_key_expr.terms) == 1:
            clear = clear_key_expr
        else:
            ck_key = expr_key(clear_key_expr)
            if ck_key not in clear_key_cache:
                clear_key_cache[ck_key] = persist(clear_key_expr)
            clear = clear_key_cache[ck_key]
        ky = ky - clear * BIG
    return [kx, ky]


def to_2d_query(q):
    """Map 1D query to 2D query expressions (purely linear)."""
    return [q, Expression({one: 1})]


def fetch(value, query=None, key=None, clear_key=None, tie_break="latest"):
    is_list = isinstance(value, (list, tuple))
    values = list(value) if is_list else [value]
    value_exprs = [to_expr(v) for v in values]
    q = to_expr(query) if query is not None else Expression()
    k = to_expr(key) if key is not None else Expression()
    ck = to_expr(clear_key) if clear_key is not None else None
    key_2d = to_2d_key(k, ck, tie_break=tie_break)
    query_2d = to_2d_query(q)
    lookup = LookUp(value_exprs, query_2d, key_2d, tie_break=tie_break)
    if is_list:
        return tuple(lookup.dims)
    return lookup.dims[0]


def name_expr_dims(name, expr):
    """Name ReGLU, Persist and LookUp dims embedded in an Expression."""
    if not isinstance(expr, Expression):
        return
    pos_idx = neg_idx = lu_idx = persist_idx = 0
    for dim, coeff in expr.terms.items():
        if isinstance(dim, PersistDimension) and dim.name.startswith("persist_"):
            dim.name = name if persist_idx == 0 else f"{name}${persist_idx}"
            persist_idx += 1
        elif isinstance(dim, ReGLUDimension) and dim.name.startswith("reglu_"):
            if coeff > 0:
                dim.name = f"{name}+" if pos_idx == 0 else f"{name}+{pos_idx}"
                pos_idx += 1
            else:
                dim.name = f"{name}-" if neg_idx == 0 else f"{name}-{neg_idx}"
                neg_idx += 1
        elif isinstance(dim, LookUpDimension) and dim.name.startswith("lookup_"):
            dim.name = f"{name}lu" if lu_idx == 0 else f"{name}lu{lu_idx}"
            if dim.lookup.name is None:
                dim.lookup.name = dim.name
            lu_idx += 1


def auto_name(local_vars):
    """Name graph nodes based on variable names from caller's locals().

    Call as auto_name(locals()) at the end of build() to give meaningful
    names to LookUps, ReGLUDimensions, etc. for diagnostic output.
    """
    for name, val in local_vars.items():
        if name.startswith("_"):
            continue
        if isinstance(val, Dimension) and not isinstance(val, InputDimension):
            val.name = name
            if isinstance(val, LookUpDimension) and val.lookup.name is None:
                val.lookup.name = name
        elif isinstance(val, Expression):
            name_expr_dims(name, val)
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, Dimension) and not isinstance(item, InputDimension):
                    item.name = f"{name}[{i}]"
                    if isinstance(item, LookUpDimension) and item.lookup.name is None:
                        item.lookup.name = name
                elif isinstance(item, Expression):
                    name_expr_dims(f"{name}[{i}]", item)


# ── Graph lifecycle ──────────────────────────────────────────────────


def reset_graph():
    """Reset all graph state for building a new program.

    Must be called before constructing a new computation graph.
    Recreates the built-in one dimension that every ALM requires.
    """
    global one
    graph_dims.clear()
    graph_lookups.clear()
    graph_memory_reads.clear()
    multiply_cache.clear()
    reglu_cache.clear()
    stepglu_cache.clear()
    clear_key_cache.clear()
    Dimension.counter = 0
    LookUp.counter = 0
    MemoryRead.counter = 0
    one = InputDimension("one")


class ProgramGraph:
    """Captured computation graph for a program, ready for scheduling
    and weight construction.

    Created by Program.build() after the graph is fully constructed.
    """

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.all_dims = list(graph_dims)
        self.all_lookups = list(graph_lookups)
        self.all_memory_reads = list(graph_memory_reads)
        self.one = one


def project_program_graph(
    program_graph: ProgramGraph, output_tokens, *, retain_input_tokens: bool = True,
) -> ProgramGraph:
    """Return the dependency closure of a new output-token set.

    Folding deliberately constructs a proof graph with more named readouts
    than an execution model needs.  Replacing an output expression alone is
    insufficient because schedulers consume ``all_dims``/``all_lookups``.
    This projection removes every operation outside the input/output closure,
    so receipt elimination happens before scheduling and weight construction.
    """
    live_dims = {program_graph.one}
    live_lookups = set()
    live_memory_reads = set()
    pending = []

    def seed(value):
        if isinstance(value, Expression):
            pending.extend(value.terms)
        elif isinstance(value, (list, tuple)):
            for item in value:
                seed(item)

    if retain_input_tokens:
        for value in program_graph.input_tokens.values():
            seed(value)
    for value in output_tokens.values():
        seed(value)

    while pending:
        dim = pending.pop()
        if dim in live_dims:
            continue
        live_dims.add(dim)
        if isinstance(dim, PersistDimension):
            pending.extend(dim.expr.terms)
        elif isinstance(dim, ReGLUDimension):
            pending.extend(dim.a_expr.terms)
            pending.extend(dim.b_expr.terms)
        elif isinstance(dim, LookUpDimension):
            lookup = dim.lookup
            if lookup in live_lookups:
                continue
            live_lookups.add(lookup)
            pending.extend(lookup.dims)
            for expr in (
                lookup.query_exprs_2d
                + lookup.key_exprs_2d
                + lookup.value_exprs
            ):
                pending.extend(expr.terms)
        elif isinstance(dim, MemoryReadDimension):
            memory_read = dim.memory_read
            if memory_read in live_memory_reads:
                continue
            live_memory_reads.add(memory_read)
            pending.extend(memory_read.dims)
            pending.extend(memory_read.address.terms)

    input_tokens = program_graph.input_tokens if retain_input_tokens else {}
    projected = ProgramGraph(input_tokens, output_tokens)
    projected.all_dims = [d for d in program_graph.all_dims if d in live_dims]
    projected.all_lookups = [
        lu for lu in program_graph.all_lookups if lu in live_lookups
    ]
    projected.all_memory_reads = [
        read
        for read in program_graph.all_memory_reads
        if read in live_memory_reads
    ]
    projected.one = program_graph.one
    return projected
