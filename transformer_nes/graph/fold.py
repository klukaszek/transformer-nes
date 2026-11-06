"""Partial evaluation of a ProgramGraph into a rebuilt (folded) graph.

``GraphFolder.fold(pins)`` pins gate persists to constants and rebuilds
only what stays live: a ReGLU with a non-positive folded gate is removed,
one with a positive constant gate inlines as a linear map, constant
persists inline, and backward liveness from the output tokens plus
``root_names`` keeps the rest.  Folded-away roots come back as named
persists so readout slots always resolve by name.

Surviving ops keep their names and memory-group annotations. A lookup whose
insert-side expressions can reach a pin is reported in
``FoldResult.insert_side_hazards`` for the caller to refuse.

The optional ``inline_linear`` construction substitutes pure linear
persists into their consumers to exchange row width for dependency depth.
Lookup operands and annotated memory reads remain materialized so
exact-structure classification and fused-read rebuilding keep their
structural licenses.

Materialization uses its own scoped GraphContext, leaving the source graph
and any surrounding construction intact.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Dimension,
    Expression,
    InputDimension,
    LookUp,
    LookUpDimension,
    MemoryRead,
    MemoryReadDimension,
    PersistDimension,
    ProgramGraph,
    ReGLUDimension,
    construction,
)
from transformer_nes.graph.evaluate import Evaluation, evaluator


def linear_input_form(expr: Expression) -> dict:
    """Expand linear persists, leaving nonlinear and memory dimensions as leaves."""
    memo = {}

    def expand_dim(dim):
        if dim in memo:
            return memo[dim]
        result = (
            expand_expr(dim.expr)
            if isinstance(dim, PersistDimension)
            else {dim: 1.0}
        )
        memo[dim] = result
        return result

    def expand_expr(value):
        result = {}
        for dim, coeff in value.terms.items():
            for source, source_coeff in expand_dim(dim).items():
                result[source] = result.get(source, 0.0) + coeff * source_coeff
        return {dim: coeff for dim, coeff in result.items() if coeff}

    return expand_expr(expr)


def const_of(expr: Expression, one: Dimension) -> float | None:
    """The float value if expr is constant (only ``one`` terms), else None."""
    for d in expr.terms:
        if d is not one:
            return None
    return expr.terms.get(one, 0.0)


def scaled_sum(pairs) -> Expression:
    """The sum of expr * c over (expr, c) pairs, accumulated term by term as
    adding them up in order would: the same rounding, term order and
    dropped zeros, without copying the running sum per term."""
    acc: dict = {}
    for expr, c in pairs:
        for e, k in expr.terms.items():
            k = k * c
            if k == 0:
                continue
            s = acc.get(e, 0) + k
            if s == 0:
                acc.pop(e, None)
            else:
                acc[e] = s
    return Expression(acc)


def subst(expr: Expression, val: dict) -> Expression:
    """Substitute folded dims; the result references only base (kept) dims."""
    return scaled_sum((Expression({d: 1}) if (v := val.get(d)) is None else v, c)
                      for d, c in expr.terms.items())


def walk_live(
    seeds, val, folded, lookup_exprs, is_leaf, bit_leaf=None,
    structural_reads: bool = False,
) -> set:
    """Backward liveness over folded union dims.  A memory-read persist
    walks only its raw ingredients (its group fabric dies); a live lookup
    produces all of its value dims.  ``bit_leaf(d)`` returns the read-site
    persist when d is a bit persist the site's fused read will serve --
    then only the site stays live and the ladder circuit dies."""
    live: set = set()
    stack = [d for d in seeds if d not in val and not is_leaf(d)]
    while stack:
        d = stack.pop()
        if d in live:
            continue
        live.add(d)
        deps: set = set()
        if isinstance(d, ReGLUDimension):
            a, b = folded[d]
            deps = set(a.terms) | set(b.terms)
        elif isinstance(d, PersistDimension):
            site = getattr(d, "memory_read", None)
            served_site = bit_leaf(d) if bit_leaf is not None else None
            if site is not None:
                keys = (
                    ("address",)
                    if structural_reads
                    else ("address", "byte_value", "write_addr", "is_write")
                )
                for key in keys:
                    deps |= set(subst(site[key], val).terms)
            elif served_site is not None:
                deps = {served_site}
            else:
                (e,) = folded[d]
                deps = set(e.terms)
                guard = getattr(d, "zero_when", None)
                if guard is not None:
                    deps |= set(subst(guard, val).terms)
        elif isinstance(d, LookUpDimension):
            fx = lookup_exprs[d.lookup]
            for e in fx["query"] + fx["key"] + fx["values"]:
                deps |= set(e.terms)
            for sib in d.lookup.dims:
                if sib not in live and sib not in val:
                    live.add(sib)
        elif isinstance(d, MemoryReadDimension):
            deps = set(d.memory_read.address.terms)
            for sibling in d.memory_read.dims:
                if sibling not in live and sibling not in val:
                    live.add(sibling)
        for dep in deps:
            if dep not in val and not is_leaf(dep) and dep not in live:
                stack.append(dep)
    return live


class GraphMaterializer:
    """Translate live folded union dims into the fresh graph context.
    Pinning hazards accumulate across per-step calls.
    Lookup deduplication is per call, so each
    sequence step gets its own lookup instances."""

    def __init__(self, structural_reads: bool = False):
        self.structural_reads = structural_reads
        self.insert_side_hazards: list[str] = []

    def materialize(self, all_dims, live, val, folded, lookup_exprs, sub, tr,
                    *, name_of=lambda n: n, is_leaf=None,
                    hazards=frozenset()) -> None:
        """Rebuild every live, not-constant dim of `all_dims` (in construction
        order) in the current graph context, in order: ReGLUs and persists from their `folded`
        operands, lookups from `lookup_exprs`, each translated by `tr` and
        recorded in `sub` (old dim -> new expression).  Annotated memory
        reads come back as one fused read; `is_leaf` dims stay as they are."""
        made = set()
        fused_reads: dict = {}
        for d in all_dims:
            if d not in live or d in val:
                continue
            if isinstance(d, InputDimension) or (is_leaf is not None and is_leaf(d)):
                continue
            if isinstance(d, ReGLUDimension):
                a, b = folded[d]
                expr = graph.reglu(tr(a), tr(b))
                next(iter(expr.terms)).name = name_of(d.name)
                sub[d] = expr
            elif isinstance(d, PersistDimension):
                site = getattr(d, "memory_read", None)
                if site is not None:
                    wanted_bits = tuple(
                        j for j, bit in enumerate(getattr(d, "read_bit_dims", ()))
                        if self.structural_reads and bit in live
                    )
                    # Translate the site's ingredients as liveness walked
                    # them -- after substitution -- so a term that cancels
                    # under the pins never reaches a dead dimension.
                    def ingredient(key, site=site):
                        return tr(subst(site[key], val))

                    if self.structural_reads:
                        expr = masked_exact_read(
                            ingredient("address"), Expression(), Expression(),
                            Expression(), structural=True,
                            bit_indices=wanted_bits,
                        )
                    else:
                        expr = masked_exact_read(
                            ingredient("address"), ingredient("byte_value"),
                            ingredient("write_addr"), ingredient("is_write"),
                        )
                    if wanted_bits:
                        read = next(iter(expr.terms)).memory_read
                        fused_reads[d] = {j: read.dims[2 + i] for i, j in enumerate(wanted_bits)}
                    nd = PersistDimension(expr, name=name_of(d.name))
                    sub[d] = Expression({nd: 1})
                    continue
                rb = getattr(d, "read_bit", None)
                if rb is not None and rb[0] in fused_reads:
                    # This bit's read serves it directly: substitute the
                    # fused lookup's bit value; the ladder never builds.
                    bit_dims = fused_reads[rb[0]]
                    sub[d] = Expression({bit_dims[rb[1]]: 1})
                    continue
                (e,) = folded[d]
                nd = PersistDimension(tr(e), name=name_of(d.name))
                guard = getattr(d, "zero_when", None)
                if guard is not None:
                    nd.zero_when = tr(subst(guard, val))
                sub[d] = Expression({nd: 1})
            elif isinstance(d, LookUpDimension):
                lu = d.lookup
                if lu in made:
                    continue
                made.add(lu)
                fx = lookup_exprs[lu]
                if self.structural_reads and lu.state_fetch:
                    # The structural backend embeds the latest state, so this
                    # fetch is an identity: values substitute directly and no
                    # head or register materializes.
                    for ud, ve in zip(lu.dims, fx["values"], strict=True):
                        sub[ud] = tr(ve)
                    continue
                nlu = LookUp(
                    [tr(e) for e in fx["values"]],
                    [tr(e) for e in fx["query"]],
                    [tr(e) for e in fx["key"]],
                    tie_break=lu.tie_break,
                )
                nlu.name = name_of(lu.name)
                nlu.memory_group = lu.memory_group
                for ud, ndim in zip(lu.dims, nlu.dims, strict=True):
                    ndim.name = name_of(ud.name)
                    sub[ud] = Expression({ndim: 1})
                if lu in hazards:
                    self.insert_side_hazards.append(lu.name or f"lookup_{lu.id}")


def masked_exact_read(addr, byte_value, write_addr, is_write, *,
                      structural: bool = False, bit_indices=()) -> Expression:
    """Rebuild an annotated memory site as one flat, exact-address lookup.
    It returns the same value as the original grouped attention: only an
    exact hit survives the mask. Grouping bounds the exported artifact's
    floating-point scores; structural execution reads the byte image directly.

    The mask is depth-lean: for integer diff, [diff > 0] is
    relu(diff) - relu(diff - 1), so the exact indicator is a LINEAR
    combination of four ReGLUs (no intermediate persists) and the gated
    result is one ReGLU -- three layers past the lookup instead of the
    seven the abs/step/multiply chain cost, on every window read's
    critical path.

    ``structural`` targets backends that serve the fused read exactly
    (value at addr when written and in range, else 0)."""
    if structural:
        bits = tuple(f"bit{7 - i}" for i in bit_indices)
        read = MemoryRead(addr, bits=bits)
        return Expression({read.dims[0]: 1})
    value, found = graph.fetch([byte_value, write_addr], query=addr,
                               key=write_addr, clear_key=1 - is_write,
                               tie_break="latest")
    value.lookup.memory_group = graph.MEMORY_GROUP_FUSED
    diff = addr - Expression({found: 1})
    exact = (Expression({graph.current_graph().one: 1})
             - graph.reglu(1, diff) + graph.reglu(1, diff - 1)
             - graph.reglu(1, -diff) + graph.reglu(1, -diff - 1))
    return graph.reglu(Expression({value: 1}), exact)


@dataclass
class FoldResult:
    """A rebuilt graph, named roots and checked pinning hazards."""

    label: str
    program_graph: ProgramGraph
    # First-seen name -> dim in the rebuilt graph.
    dim_of_name: dict[str, Dimension]
    # Union lookup names whose insert-side expressions can reach a pinned
    # gate: inserting through such a lookup would put pin-dependent history
    # into state shared across experts.  Empty for the 6502 decode pins.
    insert_side_hazards: tuple[str, ...]


class GraphFolder:
    """Folds one union ProgramGraph under different pin assignments.

    Build once per union graph; ``fold()`` per expert.  ``root_names`` are
    dim names that must exist in every folded graph (the runtime readout
    contract -- pass ``RUNTIME_READ_DIMS`` for the 6502 machine).
    """

    def __init__(self, program_graph: ProgramGraph, root_names=(), *,
                 structural_reads: bool = False):
        self.pg = program_graph
        self.root_names = tuple(root_names)
        # Structural execution folds retain only the address/read facts the
        # generated engine consumes; ordinary folds retain full lookup rows.
        self.structural_reads = structural_reads
        self.by_name: dict[str, Dimension] = {}
        for d in program_graph.all_dims:
            self.by_name.setdefault(d.name, d)
        for name in self.root_names:
            if name not in self.by_name:
                raise KeyError(f"root dim {name!r} not in the graph")
        self.evaluator = evaluator(program_graph, structural_reads)
        self.inputs = [d for d in program_graph.all_dims if isinstance(d, InputDimension)]
        self.hazard_cache = self.evaluator.hazards  # shared by every folder over the graph

    def in_order(self, dims) -> list:
        """`dims` in construction order."""
        return sorted(dims, key=self.evaluator.column.__getitem__)

    # ── Partial evaluation (graph/evaluate.py) ───────────────────────

    def evaluate(self, pins: dict[str, float], frozen: frozenset = frozenset(), *,
                 inline_linear: bool = False, preserve: frozenset = frozenset()) -> Evaluation:
        """The graph evaluated as far as `pins` allow: `val` maps every dim that became a
        constant or a linear map of others to that expression; `folded`
        holds the rest with their substituted operands (a ReGLU's (a, b), a
        persist's (expr,)); `lookup_exprs` gives each lookup's substituted
        query, key and value expressions.  `frozen` persists stay opaque
        leaves, and `inline_linear` substitutes linear persists into their
        consumers (never `preserve`d or annotated reads)."""
        return self.evaluator.evaluate(
            pins, frozenset(frozen), inline_linear=inline_linear, preserve=frozenset(preserve),
        )

    # ── Shared-state hazard: pins reaching insert-side expressions ───

    def insert_hazards(self, pin_names: frozenset[str]) -> set:
        """Union LookUps whose key/value expressions transitively read a
        pinnable gate.  Over-approximate on purpose (reachability over the
        *unfolded* union exprs, queries of upstream lookups included): a
        flagged lookup needs an explicit safety argument, not silence."""
        cached = self.hazard_cache.get(pin_names)
        if cached is not None:
            return cached
        reaches: dict = {}
        for d in self.pg.all_dims:
            if isinstance(d, InputDimension):
                reaches[d] = False
            elif isinstance(d, PersistDimension):
                reaches[d] = d.name in pin_names or any(
                    reaches.get(t, False) for t in d.expr.terms
                )
            elif isinstance(d, ReGLUDimension):
                reaches[d] = any(
                    reaches.get(t, False)
                    for t in (*d.a_expr.terms, *d.b_expr.terms)
                )
            elif isinstance(d, LookUpDimension):
                lu = d.lookup
                reaches[d] = any(
                    reaches.get(t, False)
                    for e in (lu.query_exprs + lu.key_exprs + lu.value_exprs)
                    for t in e.terms
                )
        hazards = {
            lu
            for lu in self.pg.all_lookups
            if any(
                reaches.get(t, False)
                for e in (lu.key_exprs + lu.value_exprs)
                for t in e.terms
            )
        }
        self.hazard_cache[pin_names] = hazards
        return hazards

    # ── Fold ──────────────────────────────────────────────────────────

    @construction
    def fold(self, pins: dict[str, float], label: str = "", *,
             inline_linear: bool = False) -> FoldResult:
        """The graph with `pins` (persist name -> constant) held: partial
        evaluation, liveness from the output tokens and the roots, then the
        live remainder materialized as a new ProgramGraph."""
        pg = self.pg
        pins = dict(pins)
        val, folded, lookup_exprs = self.evaluate(pins, inline_linear=inline_linear)
        hazards = self.insert_hazards(frozenset(pins))
        # Folded output-token expressions (they seed liveness and become the
        # new graph's output tokens).  Root dims are exempt from the output
        # substitution: a folded-away root is re-materialized below as a
        # named persist, and the output tokens must reference that persist
        # (keeping it alive to the final layer), not its inlined expansion.
        root_dims = [self.by_name[n] for n in self.root_names]
        val_out = {d: v for d, v in val.items() if d not in set(root_dims)}
        fout = {
            name: subst(expr, val_out)
            for name, expr in pg.output_tokens.items()
            if isinstance(expr, Expression)
        }

        # Liveness seeds: output-token terms plus the root dims -- a root
        # that folded away seeds with its folded expression's terms instead.
        seeds: set = set()
        for e in fout.values():
            seeds |= set(e.terms)
        for d in root_dims:
            seeds |= set(val[d].terms) if d in val else {d}

        def bit_leaf(d):
            rb = getattr(d, "read_bit", None)
            return rb[0] if rb is not None and self.structural_reads else None

        live = walk_live(
            seeds, val, folded, lookup_exprs,
            lambda d: isinstance(d, InputDimension),
            bit_leaf=bit_leaf,
            structural_reads=self.structural_reads,
        )

        # ── Materialize the new graph ─────────────────────────────────
        sub: dict = {
            pg.one: Expression({graph.current_graph().one: 1}),
        }
        for d in pg.all_dims:
            if isinstance(d, InputDimension) and d not in sub:
                sub[d] = Expression({InputDimension(d.name): 1})

        def tr(expr: Expression) -> Expression:
            def term(d):
                s = sub.get(d)
                if s is None:
                    if d not in val:
                        raise KeyError(f"fold({label}): {d!r} referenced but neither live nor folded")
                    s = sub[d] = tr(val[d])
                return s
            return scaled_sum((term(d), c) for d, c in expr.terms.items())

        mat = GraphMaterializer(structural_reads=self.structural_reads)
        mat.materialize(self.in_order(live), live, val, folded, lookup_exprs, sub, tr,
                        hazards=hazards)

        # Folded-away roots come back as named persists so readout slots
        # always resolve by name (and stay alive via the output tokens).
        for d in root_dims:
            if d in val:
                nd = PersistDimension(tr(val[d]), name=d.name)
                sub[d] = Expression({nd: 1})

        new_inputs = {
            name: tr(expr) if isinstance(expr, Expression) else expr
            for name, expr in pg.input_tokens.items()
        }
        new_outputs = {
            name: tr(fout[name]) if name in fout else expr
            for name, expr in pg.output_tokens.items()
        }
        new_pg = graph.current_graph().capture(new_inputs, new_outputs)

        dim_of_name: dict[str, Dimension] = {}
        for d in new_pg.all_dims:
            dim_of_name.setdefault(d.name, d)

        return FoldResult(
            label=label,
            program_graph=new_pg,
            dim_of_name=dim_of_name,
            insert_side_hazards=tuple(mat.insert_side_hazards),
        )
