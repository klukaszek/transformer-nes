"""Partial evaluation of a ProgramGraph into a rebuilt (folded) graph.

``GraphFolder.fold(pins)`` pins gate persists to constants and rebuilds
only what stays live: a ReGLU with a non-positive folded gate is removed,
one with a positive constant gate inlines as a linear map, constant
persists inline, and backward liveness from the output tokens plus
``root_names`` keeps the rest.  Folded-away roots come back as named
persists so readout slots always resolve by name.

Surviving ops keep their union names, and LookUps carry provenance
(``union_lookup_of``, ``memory_group``) so engines share attention state
across folds.  That contract is checked, not assumed: a lookup whose
insert-side expressions can reach a pin is reported in
``FoldResult.insert_side_hazards`` for the caller to refuse.

The optional ``inline_linear`` construction substitutes pure linear
persists into their consumers to exchange row width for dependency depth.
Lookup operands and annotated memory reads remain materialized so
exact-structure classification and fused-read rebuilding keep their
structural licenses.

NOTE: materialization calls ``reset_graph()``, so the module-level graph
context afterwards holds the *folded* graph; the union survives through
the folder's ProgramGraph snapshot, and any later construction must call
``reset_graph()`` itself (build() does).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    reset_graph,
    stepglu,
)


class SequenceRejection(RuntimeError):
    """A sequence/trace fold license failure at a known step, so callers
    can retry with the window truncated before it."""

    def __init__(self, msg: str, step: int | None = None):
        super().__init__(msg)
        self.step = step


def const_of(expr: Expression, one: Dimension) -> float | None:
    """The float value if expr is constant (only ``one`` terms), else None."""
    for d in expr.terms:
        if d is not one:
            return None
    return expr.terms.get(one, 0.0)


def subst(expr: Expression, val: dict) -> Expression:
    """Substitute folded dims; the result references only base (kept) dims."""
    out = Expression()
    for d, c in expr.terms.items():
        v = val.get(d)
        out = out + (Expression({d: c}) if v is None else v * c)
    return out


def graph_delta(start_d: int, start_l: int, start_m: int) -> tuple[int, int, int, int]:
    """(reglus, persists, lookups, terms) created since the registry marks;
    the fused-read builder goes through the DSL, so its dims are counted
    from the registry rather than by hand."""
    nr = np_ = nl = nt = 0
    for x in graph.graph_dims[start_d:]:
        if isinstance(x, ReGLUDimension):
            nr += 1
            nt += len(x.a_expr.terms) + len(x.b_expr.terms)
        elif isinstance(x, PersistDimension):
            np_ += 1
            nt += len(x.expr.terms)
    for lu in graph.graph_lookups[start_l:]:
        nl += 1
        for e in lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs:
            nt += len(e.terms)
    for read in graph.graph_memory_reads[start_m:]:
        nl += 1
        nt += len(read.address.terms)
    return nr, np_, nl, nt


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
    Counters and lookup provenance accumulate across per-step calls
    (lookup_of_union keeps the last instance); the same-lookup dedup is
    per call, so each sequence step gets its own lookup instances."""

    def __init__(self, structural_reads: bool = False):
        self.structural_reads = structural_reads
        self.union_lookup_of: dict[int, int] = {}
        self.lookup_of_union: dict[int, LookUp] = {}
        self.insert_side_hazards: list[str] = []
        self.n_reglus = 0
        self.n_persists = 0
        self.n_lookups = 0
        self.n_terms = 0

    def materialize(self, all_dims, live, val, folded, lookup_exprs, sub, tr,
                    *, name_of=lambda n: n, is_leaf=None,
                    hazards=frozenset(), resolve_read=None) -> None:
        """Rebuild every live, not-constant dim of `all_dims` in the current
        graph context, in order: ReGLUs and persists from their `folded`
        operands, lookups from `lookup_exprs`, each translated by `tr` and
        recorded in `sub` (old dim -> new expression).  Annotated memory
        reads come back as one fused read (or a forwarded store when
        `resolve_read` finds one); `is_leaf` dims stay as they are."""
        made = set()
        fused_reads: dict = {}
        for d in all_dims:
            if d in val or d not in live:
                continue
            if isinstance(d, InputDimension) or (is_leaf is not None and is_leaf(d)):
                continue
            if isinstance(d, ReGLUDimension):
                a, b = folded[d]
                nd = ReGLUDimension(tr(a), tr(b), name=name_of(d.name))
                sub[d] = Expression({nd: 1})
                self.n_reglus += 1
                self.n_terms += len(nd.a_expr.terms) + len(nd.b_expr.terms)
            elif isinstance(d, PersistDimension):
                site = getattr(d, "memory_read", None)
                if site is not None:
                    fwd = resolve_read(d, site, tr) if resolve_read else None
                    if fwd is not None:
                        # Store-to-load forwarding: an in-window write at a
                        # provably identical address supplies the value; the
                        # lookup fabric never materializes.
                        nd = PersistDimension(fwd, name=name_of(d.name))
                        sub[d] = Expression({nd: 1})
                        self.n_persists += 1
                        self.n_terms += len(nd.expr.terms)
                        continue
                    marks = (
                        len(graph.graph_dims), len(graph.graph_lookups),
                        len(graph.graph_memory_reads),
                    )
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
                        fused_reads[d] = (
                            read,
                            {j: read.dims[2 + i] for i, j in enumerate(wanted_bits)},
                        )
                    nd = PersistDimension(expr, name=name_of(d.name))
                    sub[d] = Expression({nd: 1})
                    dr, dp, dl, dt = graph_delta(*marks)
                    self.n_reglus += dr
                    self.n_persists += dp
                    self.n_lookups += dl
                    self.n_terms += dt
                    continue
                rb = getattr(d, "read_bit", None)
                if rb is not None and rb[0] in fused_reads:
                    # This bit's read serves it directly: substitute the
                    # fused lookup's bit value; the ladder never builds.
                    _read, bit_dims = fused_reads[rb[0]]
                    sub[d] = Expression({bit_dims[rb[1]]: 1})
                    continue
                (e,) = folded[d]
                nd = PersistDimension(tr(e), name=name_of(d.name))
                sub[d] = Expression({nd: 1})
                self.n_persists += 1
                self.n_terms += len(nd.expr.terms)
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
                self.union_lookup_of[nlu.id] = lu.id
                self.lookup_of_union[lu.id] = nlu
                for ud, ndim in zip(lu.dims, nlu.dims, strict=True):
                    ndim.name = name_of(ud.name)
                    sub[ud] = Expression({ndim: 1})
                if lu in hazards:
                    self.insert_side_hazards.append(lu.name or f"lookup_{lu.id}")
                self.n_lookups += 1
                for e in nlu.query_exprs_2d + nlu.key_exprs_2d + nlu.value_exprs:
                    self.n_terms += len(e.terms)


EXECUTION_INTERIOR_ROOTS = frozenset({
    "ea", "next_pc", "new_halted", "cycles",
    "is_store", "write_addr_out", "write_value_out",
    "is_store2", "write_addr2_out", "write_value2_out",
    "is_store3", "write_addr3_out", "write_value3_out",
})


def masked_exact_read(addr, byte_value, write_addr, is_write, *,
                      structural: bool = False, bit_indices=()) -> Expression:
    """A memory_at site rebuilt flat: one 64K-span lookup plus the
    exact-address mask.  Identical output to the union's 16-group sum
    (the mask keeps exact hits only, and an exact hit answers from its
    own group either way); the group split exists for the exported
    artifact's f32 score range, which expert engines never compute.

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
    exact = (Expression({graph.one: 1})
             - graph.reglu(1, diff) + graph.reglu(1, diff - 1)
             - graph.reglu(1, -diff) + graph.reglu(1, -diff - 1))
    return graph.reglu(Expression({value: 1}), exact)


@dataclass
class FoldResult:
    """A rebuilt graph plus the provenance an engine needs to share state."""

    label: str
    pins: dict[str, float]
    program_graph: ProgramGraph
    # First-seen name -> dim in the *new* graph (mirrors dim_id_by_name).
    dim_of_name: dict[str, Dimension]
    # New LookUp id -> union LookUp id (and the reverse, by union id).
    union_lookup_of: dict[int, int]
    lookup_of_union: dict[int, LookUp]
    # Union lookup names whose insert-side expressions can reach a pinned
    # gate: inserting through such a lookup would put pin-dependent history
    # into state shared across experts.  Empty for the 6502 decode pins.
    insert_side_hazards: tuple[str, ...]
    n_reglus: int = 0
    n_persists: int = 0
    n_lookups: int = 0
    n_terms: int = 0
    # Root names that folded away and were re-materialized as persists.
    folded_roots: tuple[str, ...] = field(default=())
    # Sequence folds: number of composed steps (root/readout names carry a
    # __s<k> suffix per step; step-0 input dims keep their canonical names).
    n_steps: int = 1
    # forward_stores folds: one (step, union read name, const address |
    # None, forwarded) row per surviving memory read, for read licenses.
    read_log: tuple = field(default=())


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
        # The generated engine classifies lookup rows structurally.
        # Keep direct linear lookup operands as residual coordinates rather
        # than expanding (for example) an exact clear-gate coordinate back
        # into ``1e30 * is_state - 1e30 * one``.
        self.lookup_persists = {
            d
            for lu in program_graph.all_lookups
            for e in (lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs)
            for d in e.terms
            if isinstance(d, PersistDimension)
        }
        self.by_name: dict[str, Dimension] = {}
        for d in program_graph.all_dims:
            self.by_name.setdefault(d.name, d)
        for name in self.root_names:
            if name not in self.by_name:
                raise KeyError(f"root dim {name!r} not in the graph")
        self.hazard_cache: dict[frozenset[str], set] = {}
        self.dependency_cache: dict[frozenset[Dimension], frozenset[Dimension]] = {}

    def dependency_closure(self, seeds) -> frozenset[Dimension]:
        """Every dim `seeds` read, transitively (memoized).  With
        structural reads, a memory read depends on its address only, and a
        served bit on its read."""
        key = frozenset(seeds)
        if key in self.dependency_cache:
            return self.dependency_cache[key]
        live = set(key)
        pending = list(key)
        while pending:
            dim = pending.pop()
            deps = set()
            if isinstance(dim, PersistDimension):
                site = getattr(dim, "memory_read", None)
                read_bit = getattr(dim, "read_bit", None)
                if site is not None and self.structural_reads:
                    deps.update(site["address"].terms)
                elif read_bit is not None and self.structural_reads:
                    deps.add(read_bit[0])
                else:
                    deps.update(dim.expr.terms)
            elif isinstance(dim, ReGLUDimension):
                deps.update(dim.a_expr.terms)
                deps.update(dim.b_expr.terms)
            elif isinstance(dim, LookUpDimension):
                lookup = dim.lookup
                deps.update(lookup.dims)
                for expr in (
                    lookup.query_exprs_2d
                    + lookup.key_exprs_2d
                    + lookup.value_exprs
                ):
                    deps.update(expr.terms)
            for dep in deps:
                if dep not in live:
                    live.add(dep)
                    pending.append(dep)
        result = frozenset(live)
        self.dependency_cache[key] = result
        return result

    # ── Partial evaluation ───────────────────────────────────────────

    def partial_eval(self, pins: dict[str, float], frozen: frozenset = frozenset(),
                     *, inline_linear: bool = False,
                     preserve: frozenset = frozenset(),
                     resolve_site=None, inline_budget: int | None = None,
                     required: frozenset[Dimension] | None = None):
        """Evaluate the graph as far as `pins` allow, in construction order.

        Returns (val, folded, lookup_exprs): `val` maps every dim that
        became a constant or a linear map of others to that expression;
        `folded` holds the rest with their substituted operands (a ReGLU's
        (a, b), a persist's (expr,)); `lookup_exprs` gives each lookup's
        substituted query, key and value expressions.  `frozen` persists
        stay opaque leaves, `required` limits the walk to those dims, and
        `inline_linear` substitutes linear persists into their consumers
        (within `inline_budget` terms, never `preserve`d or annotated
        reads)."""
        pg = self.pg
        one = pg.one
        val: dict = {one: Expression({one: 1.0})}
        folded: dict = {}
        for d in pg.all_dims:
            if required is not None and d not in required:
                continue
            if isinstance(d, PersistDimension) and d.name in frozen:
                # Opaque leaf: bound to another step's output at
                # materialization (fold_sequence), never folded through.
                continue
            if isinstance(d, PersistDimension) and d.name in pins:
                val[d] = Expression({one: float(pins[d.name])})
                continue
            if resolve_site is not None and isinstance(d, PersistDimension):
                site = getattr(d, "memory_read", None)
                if site is not None:
                    c = resolve_site(d, site, val)
                    if c is not None:
                        # Constant store-to-load forward: the read IS the
                        # in-window written constant, visible to all
                        # downstream constant folding (an RTS return
                        # address must reach next_pc as a constant).
                        val[d] = Expression({one: float(c)})
                        continue
            if isinstance(d, ReGLUDimension):
                a = subst(d.a_expr, val)
                b = subst(d.b_expr, val)
                bc = const_of(b, one)
                ac = const_of(a, one)
                if (bc is not None and bc <= 0) or (ac is not None and ac == 0):
                    val[d] = Expression()
                elif bc is not None and ac is not None:
                    val[d] = Expression({one: ac * bc})
                elif bc is not None:
                    # relu(b)*a with b a known positive constant: an exact
                    # linear map, inlined into every consumer.
                    val[d] = a * bc
                else:
                    folded[d] = (a, b)
            elif isinstance(d, PersistDimension):
                mo = getattr(d, "multiply_of", None) if self.structural_reads \
                    else None
                if mo is not None:
                    # A tagged product folds to an exact linear map when
                    # either factor pins to a constant; its ReGLU pair
                    # (which only folds on a constant b) goes dead.
                    ma, mb = subst(mo[0], val), subst(mo[1], val)
                    mac, mbc = const_of(ma, one), const_of(mb, one)
                    if mac is not None:
                        val[d] = mb * mac
                        continue
                    if mbc is not None:
                        val[d] = ma * mbc
                        continue
                e = subst(d.expr, val)
                c = const_of(e, one)
                if c is not None:
                    val[d] = Expression({one: c})
                elif (
                    inline_linear
                    and d not in self.lookup_persists
                    and d not in preserve
                    and getattr(d, "memory_read", None) is None
                    and (not self.structural_reads
                         or getattr(d, "read_bit", None) is None)
                    and (inline_budget is None or len(e.terms) <= inline_budget)
                ):
                    # A persist is a pure linear map, so folds may substitute
                    # it into consumers to trade wider rows for a shorter
                    # chain; inline_budget bounds that width, and annotated
                    # memory reads stay materialized for the fused lookup.
                    val[d] = e
                else:
                    folded[d] = (e,)
        lookup_exprs: dict = {}
        for lu in pg.all_lookups:
            if required is not None and not any(d in required for d in lu.dims):
                continue
            lookup_exprs[lu] = {
                "query": [subst(e, val) for e in lu.query_exprs_2d],
                "key": [subst(e, val) for e in lu.key_exprs_2d],
                "values": [subst(e, val) for e in lu.value_exprs],
            }
        return val, folded, lookup_exprs

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
                    for e in (lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs)
                    for t in e.terms
                )
        hazards = {
            lu
            for lu in self.pg.all_lookups
            if any(
                reaches.get(t, False)
                for e in (lu.key_exprs_2d + lu.value_exprs)
                for t in e.terms
            )
        }
        self.hazard_cache[pin_names] = hazards
        return hazards

    # ── Fold ──────────────────────────────────────────────────────────

    def fold(self, pins: dict[str, float], label: str = "", *,
             inline_linear: bool = False) -> FoldResult:
        """The graph with `pins` (persist name -> constant) held: partial
        evaluation, liveness from the output tokens and the roots, then
        the live remainder materialized as a new ProgramGraph."""
        pg = self.pg
        pins = dict(pins)
        val, folded, lookup_exprs = self.partial_eval(
            pins, inline_linear=inline_linear,
        )
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
        reset_graph()
        sub: dict = {
            pg.one: Expression({graph.one: 1}),
        }
        for d in pg.all_dims:
            if isinstance(d, InputDimension) and d not in sub:
                sub[d] = Expression({InputDimension(d.name): 1})

        def tr(expr: Expression) -> Expression:
            out = Expression()
            for d, c in expr.terms.items():
                s = sub.get(d)
                if s is None:
                    if d in val:
                        s = sub[d] = tr(val[d])
                    else:
                        raise KeyError(
                            f"fold({label}): {d!r} referenced but neither live nor folded"
                        )
                out = out + s * c
            return out

        mat = GraphMaterializer(structural_reads=self.structural_reads)
        mat.materialize(pg.all_dims, live, val, folded, lookup_exprs, sub, tr,
                        hazards=hazards)

        # Folded-away roots come back as named persists so readout slots
        # always resolve by name (and stay alive via the output tokens).
        folded_roots: list[str] = []
        for d in root_dims:
            if d in val:
                nd = PersistDimension(tr(val[d]), name=d.name)
                sub[d] = Expression({nd: 1})
                folded_roots.append(d.name)
                mat.n_persists += 1
                mat.n_terms += len(nd.expr.terms)

        new_inputs = {
            name: tr(expr) if isinstance(expr, Expression) else expr
            for name, expr in pg.input_tokens.items()
        }
        new_outputs = {
            name: tr(fout[name]) if name in fout else expr
            for name, expr in pg.output_tokens.items()
        }
        new_pg = ProgramGraph(new_inputs, new_outputs)

        dim_of_name: dict[str, Dimension] = {}
        for d in new_pg.all_dims:
            dim_of_name.setdefault(d.name, d)

        return FoldResult(
            label=label,
            pins=pins,
            program_graph=new_pg,
            dim_of_name=dim_of_name,
            union_lookup_of=mat.union_lookup_of,
            lookup_of_union=mat.lookup_of_union,
            insert_side_hazards=tuple(mat.insert_side_hazards),
            n_reglus=mat.n_reglus,
            n_persists=mat.n_persists,
            n_lookups=mat.n_lookups,
            n_terms=mat.n_terms,
            folded_roots=tuple(folded_roots),
        )


    # ── Sequence fold: K chained steps as one graph ───────────────────

    def fold_sequence(self, pin_list, feedback, label: str = "",
                      keep_alive_token: str = "READOUT",
                      pc_offsets=None, *, inline_linear: bool = False,
                      forward_stores: bool = False,
                      read_bounds=None,
                      inline_budget: int | None = None,
                      bit_feedback_until: int | None = None,
                      _execution: bool = False) -> FoldResult:
        """Fold a K-step composition of the union graph into ONE graph.

        pin_list[k] pins step k (the router supplies them: pc, instruction
        bytes, is_nmi=0).  feedback maps a step's state-input persist name to
        the previous step's output name -- either "out_name" (bound as an
        expression, or as a pin when the previous fold made it constant) or
        ("out_name", "lo"|"hi") (byte split; a non-constant source leaves
        the input UNBOUND -- fine when the fold proves it dead, a loud
        KeyError when anything live still reads it).
        Every step's root dims are materialized with a ``__s<k>`` suffix and
        kept alive through ``keep_alive_token``; the other output tokens are
        substituted through the FINAL step, and step 0's input dims keep
        their canonical names (the composed graph embeds one state token).

        pc_offsets, if given, folds a canonical *shape*: step k's pc is not
        pinned but derived as head + pc_offsets[k], where head is step 0's
        symbolic (input-bound) pc.  The byte split is one stepglu carry per
        step -- exact for integer bytes -- so a single fold serves every
        head the router matches.  License (checked by the caller): the
        window must not wrap $FFFF, and intermediate control flow must
        actually be pc + offset (verify folded next_pc__sk).

        forward_stores enables store-to-load forwarding: every step's store
        roots must fold to constant flags and (when storing) constant
        addresses; a later step's memory read at a matching constant
        address takes the written value expression instead of a lookup, and
        a read at a distinct constant address keeps the global lookup (the
        engine's ordered post-forward writes cannot alter it).  Anything
        unprovable -- a non-constant store flag or address, or a
        non-constant read address while writes are pending -- raises, so
        callers treat it as a license rejection.  A non-constant read
        with pending writes is admitted only when ``read_bounds``
        (caller-certified (step, read name) -> (lo, hi) address bounds)
        proves it disjoint from every pending store.  Surviving reads are
        recorded in FoldResult.read_log for the caller's range licenses.
        """
        pg = self.pg
        one = pg.one
        n_steps = len(pin_list)
        root_dims = [self.by_name[n] for n in self.root_names]
        feedback = dict(feedback)
        # Bit-state feedback (wide_bits only): A's bits chain like A itself.
        # ``bit_feedback_until`` prunes the chain past the last step whose
        # opcode consumes A bits, so unseeded steps drop the arith ladder
        # rather than keep it live; wrong pruning fails loudly in tr().
        bit_ins, bit_outs = set(), set()
        for b in range(8):
            if f"new_a_bit{b}" in self.by_name:
                feedback.setdefault(f"current_a_bit{b}", f"new_a_bit{b}")
                bit_ins.add(f"current_a_bit{b}")
                bit_outs.add(f"new_a_bit{b}")
        for out in feedback.values():
            if isinstance(out, (int, float)):
                continue  # constant feedback (e.g. the always-1 U flag)
            out_name = out[0] if isinstance(out, tuple) else out
            if out_name not in self.by_name:
                raise KeyError(f"feedback output {out_name!r} not in the graph")

        reset_graph()
        base_sub: dict = {
            pg.one: Expression({graph.one: 1}),
        }
        for d in pg.all_dims:
            if isinstance(d, InputDimension) and d not in base_sub:
                base_sub[d] = Expression({InputDimension(d.name): 1})

        all_pins: dict[str, float] = {}
        mat = GraphMaterializer(  # counters/provenance accumulate per step
            structural_reads=self.structural_reads,
        )
        insert_side_hazards: set[str] = set()
        folded_roots: list[str] = []
        keep_alive = Expression()
        fb_expr: dict[str, Expression] = {}  # prev step's outputs, new-graph space
        fb_const: dict[str, float] = {}
        new_outputs = None
        # forward_stores state: earlier steps' (const address, const value |
        # None, new-graph value expr | None), latest last, plus the read
        # disposition log.
        pending_writes: list[tuple[int, float | None, Expression | None]] = []
        read_log: list[tuple[int, str, int | None, bool]] = []

        pc_names = ("current_pc_lo", "current_pc_hi")
        pc_exprs = None  # step-0 (head) pc byte expressions, new-graph space

        for k in range(n_steps):
            step_root_dims = (
                root_dims
                if not _execution or k + 1 == n_steps
                else [
                    self.by_name[name] for name in self.root_names
                    if name in EXECUTION_INTERIOR_ROOTS
                ]
            )
            seed_bits = bit_feedback_until is None or k < bit_feedback_until
            fb_dims = [
                self.by_name[o[0] if isinstance(o, tuple) else o]
                for o in feedback.values()
                if not isinstance(o, (int, float))
                and (seed_bits or o not in bit_outs)
            ] if k + 1 < n_steps else []
            pins = dict(pin_list[k])
            frozen: set[str] = set()
            unbound: set[str] = set()
            if pc_offsets is not None and k > 0:
                frozen |= set(pc_names)
            if k > 0:
                for in_name, out in feedback.items():
                    if in_name in pins:  # explicit pins beat feedback
                        continue
                    if isinstance(out, (int, float)):
                        pins[in_name] = float(out)
                        continue
                    if isinstance(out, tuple):
                        out_name, part = out
                        c = fb_const.get(out_name)
                        if c is None:
                            # Non-constant byte-split source: leave the
                            # input unbound.  Dead under the fold -> fine;
                            # a live reference fails loudly in tr().
                            frozen.add(in_name)
                            unbound.add(in_name)
                            continue
                        v = int(round(c))
                        pins[in_name] = float(v & 0xFF if part == "lo" else (v >> 8) & 0xFF)
                    elif (c := fb_const.get(out)) is not None:
                        pins[in_name] = c
                    else:
                        frozen.add(in_name)
                        if in_name in bit_ins and out not in fb_expr:
                            # Pruned bit feedback: leave unbound; a live
                            # reader fails loudly in tr().
                            unbound.add(in_name)
            for name, v in pins.items():
                all_pins[f"s{k}:{name}"] = v

            fwd_expr_of: dict = {}

            def resolve_site(d, site, val, *, k=k, fwd_expr_of=fwd_expr_of):
                ae = subst(site["address"], val)
                c = const_of(ae, one)
                if c is None:
                    return None  # live reads are arbitrated at materialize
                a = int(round(c)) & 0xFFFF
                for pa, pconst, pexpr in reversed(pending_writes):
                    if pa != a:
                        continue
                    if pconst is not None:
                        read_log.append((k, d.name, a, True))
                        return pconst
                    fwd_expr_of[d] = pexpr
                    return None
                return None

            val, folded, lookup_exprs = self.partial_eval(
                pins, frozenset(frozen), inline_linear=inline_linear,
                preserve=frozenset(
                    self.by_name[n] for n in pc_names
                ) if pc_offsets is not None and k == 0 else frozenset(),
                resolve_site=resolve_site if forward_stores else None,
                inline_budget=inline_budget,
                required=(
                    self.dependency_closure(step_root_dims + fb_dims)
                    if _execution else None
                ),
            )
            insert_side_hazards |= {
                lu.name or f"lookup_{lu.id}"
                for lu in self.insert_hazards(frozenset(pins))
            }

            # Liveness seeds: this step's roots, the outputs the next step
            # feeds on, and (final step) the graph's output tokens.
            fout = {}
            if k + 1 == n_steps and not _execution:
                val_out = {d: v for d, v in val.items() if d not in set(root_dims)}
                fout = {
                    name: subst(expr, val_out)
                    for name, expr in pg.output_tokens.items()
                    if isinstance(expr, Expression)
                }
            seeds: set = set()
            for e in fout.values():
                seeds |= set(e.terms)
            for d in step_root_dims + fb_dims:
                seeds |= set(val[d].terms) if d in val else {d}

            def is_leaf(d, *, frozen=frozen):
                return (
                    isinstance(d, InputDimension)
                    or (isinstance(d, PersistDimension) and d.name in frozen)
                )

            def bit_leaf(d, *, fwd_expr_of=fwd_expr_of):
                rb = getattr(d, "read_bit", None)
                if rb is None or not self.structural_reads:
                    return None
                return None if rb[0] in fwd_expr_of else rb[0]

            live = walk_live(seeds, val, folded, lookup_exprs, is_leaf,
                             bit_leaf=bit_leaf,
                             structural_reads=self.structural_reads)

            # Materialize this step (fresh sub: the same union dim holds a
            # different value per step); frozen persists resolve to the
            # previous step's already-materialized output expressions.
            sub = dict(base_sub)
            for in_name in frozen:
                if in_name in unbound or in_name in pc_names:
                    continue
                sub[self.by_name[in_name]] = fb_expr[feedback[in_name]]
            if pc_offsets is not None and k > 0:
                # Shape fold: pc(k) = head + offset, byte-split with one
                # stepglu carry (exact on integer bytes).  No $FFFF wrap by
                # the caller's license.
                off = int(pc_offsets[k])
                assert pc_exprs is not None  # captured at k == 0
                lo0, hi0 = pc_exprs
                if off == 0:
                    lo_e, hi_e = lo0, hi0
                else:
                    carry = stepglu(1, lo0 + Expression({graph.one: off - 256}))
                    lo_e = lo0 + Expression({graph.one: float(off)}) + carry * -256.0
                    hi_e = hi0 + carry
                sub[self.by_name["current_pc_lo"]] = lo_e
                sub[self.by_name["current_pc_hi"]] = hi_e

            def tr(expr: Expression, *, sub=sub, val=val, k=k) -> Expression:
                out = Expression()
                for d, c in expr.terms.items():
                    s = sub.get(d)
                    if s is None:
                        if d in val:
                            s = sub[d] = tr(val[d])
                        else:
                            raise KeyError(
                                f"fold_sequence({label}) s{k}: {d!r} neither "
                                "live nor folded"
                            )
                    out = out + s * c
                return out

            def sname(name, *, k=k):
                return f"{name}__s{k}" if name else None

            def resolve_read(d, site, tr, *, k=k, fwd_expr_of=fwd_expr_of):
                pv = fwd_expr_of.get(d)
                if pv is not None:
                    read_log.append((k, d.name, None, True))
                    return pv
                addr = tr(site["address"])
                c = const_of(addr, graph.one)
                if c is None:
                    b = read_bounds.get((k, d.name)) if read_bounds else None
                    disjoint = b is not None and all(
                        not (b[0] <= pa <= b[1])
                        for pa, _pc, _pe in pending_writes
                    )
                    if pending_writes and not disjoint:
                        raise SequenceRejection(
                            f"fold_sequence({label}) s{k}: memory read "
                            f"{d.name!r} has a non-constant address after "
                            "in-window stores", step=k,
                        )
                    read_log.append((k, d.name, None, False))
                    return None
                read_log.append((k, d.name, int(round(c)) & 0xFFFF, False))
                return None

            mat.materialize(pg.all_dims, live, val, folded, lookup_exprs,
                            sub, tr, name_of=sname, is_leaf=is_leaf,
                            resolve_read=resolve_read if forward_stores
                            else None)

            # Roots: every step's readout set, suffixed; folded-away roots
            # come back as named persists (same contract as fold()).
            for d in step_root_dims:
                if d in val:
                    nd = PersistDimension(tr(val[d]), name=sname(d.name))
                    sub[d] = Expression({nd: 1})
                    folded_roots.append(nd.name)
                    mat.n_persists += 1
                    mat.n_terms += len(nd.expr.terms)
                keep_alive = keep_alive + 1e-6 * sub[d]

            if pc_offsets is not None and k == 0 and n_steps > 1:
                pc_exprs = tuple(
                    tr(Expression({self.by_name[n]: 1})) for n in pc_names
                )

            if forward_stores:
                for flag_n, addr_n, val_n in (
                    ("is_store", "write_addr_out", "write_value_out"),
                    ("is_store2", "write_addr2_out", "write_value2_out"),
                    ("is_store3", "write_addr3_out", "write_value3_out"),
                ):
                    fe = val.get(self.by_name[flag_n])
                    fc = None if fe is None else const_of(fe, one)
                    if fc is None:
                        raise SequenceRejection(
                            f"fold_sequence({label}) s{k}: {flag_n} did not "
                            "fold to a constant", step=k,
                        )
                    if int(round(fc)) == 0:
                        continue
                    ae = val.get(self.by_name[addr_n])
                    ac = None if ae is None else const_of(ae, one)
                    if ac is None:
                        raise SequenceRejection(
                            f"fold_sequence({label}) s{k}: {addr_n} did not "
                            "fold to a constant", step=k,
                        )
                    # Constant values forward through partial_eval (an RTS
                    # return address must reach next_pc as a constant);
                    # live values forward as the already-materialized root
                    # persist, shared by every later reader.
                    ve = val.get(self.by_name[val_n])
                    vc = None if ve is None else const_of(ve, one)
                    pending_writes.append((
                        int(round(ac)) & 0xFFFF,
                        vc,
                        None if vc is not None
                        else tr(Expression({self.by_name[val_n]: 1})),
                    ))

            fb_const, fb_expr = {}, {}
            if k + 1 < n_steps:
                # The last step feeds nothing; non-root feedback outputs
                # (bit state) are not kept live there, so tr would fail.
                for out in set(feedback.values()):
                    if isinstance(out, (int, float)):
                        continue
                    out_name = out[0] if isinstance(out, tuple) else out
                    if out_name in bit_outs and not seed_bits:
                        continue
                    d = self.by_name[out_name]
                    src = val.get(d, Expression({d: 1}))
                    c = const_of(src, one)
                    if c is not None:
                        fb_const[out_name] = c
                    else:
                        fb_expr[out_name] = tr(src)

            if k + 1 == n_steps:
                new_inputs = {
                    name: tr(expr) if isinstance(expr, Expression) else expr
                    for name, expr in pg.input_tokens.items()
                }
                new_outputs = (
                    {
                        name: tr(fout[name]) if name in fout else expr
                        for name, expr in pg.output_tokens.items()
                    }
                    if not _execution else {}
                )

        assert new_inputs is not None and new_outputs is not None
        prev = new_outputs.get(keep_alive_token)
        prev = prev if isinstance(prev, Expression) else Expression()
        new_outputs[keep_alive_token] = prev + keep_alive
        new_pg = ProgramGraph(new_inputs, new_outputs)

        dim_of_name: dict[str, Dimension] = {}
        for d in new_pg.all_dims:
            dim_of_name.setdefault(d.name, d)

        return FoldResult(
            label=label,
            pins=all_pins,
            program_graph=new_pg,
            dim_of_name=dim_of_name,
            union_lookup_of=mat.union_lookup_of,
            lookup_of_union=mat.lookup_of_union,
            insert_side_hazards=tuple(sorted(insert_side_hazards)),
            n_reglus=mat.n_reglus,
            n_persists=mat.n_persists,
            n_lookups=mat.n_lookups,
            n_terms=mat.n_terms,
            folded_roots=tuple(folded_roots),
            n_steps=n_steps,
            read_log=tuple(read_log),
        )

    def fold_execution_sequence(self, pin_list, feedback, **kwargs) -> FoldResult:
        return self.fold_sequence(
            pin_list, feedback, _execution=True, **kwargs,
        )


def opcode_pins(byte: int, opcodes, *, pin_fetched_byte: bool = True) -> dict[str, float]:
    """Pin every decode gate for a single-opcode expert: ``is_op_<byte>=1``,
    every other ``is_op_*=0``, ``is_nmi=0`` (an NMI overrides the fetched
    instruction, so a routed step must not claim it).

    With ``pin_fetched_byte`` (the default), the ``fetched_opcode`` persist
    itself is pinned to the byte.  The router dispatches on
    ``ram[pc] == byte``, which is strictly stronger than ``is_op = 1``, so
    the memory read at PC provably returns the byte -- pinning it folds
    that read (one of the three grouped fetch reads) and any remaining
    decode residue out of the expert graph."""
    pins: dict[str, float] = {"is_nmi": 0.0, "is_irq": 0.0}
    for op in opcodes.values():
        pins[f"is_op_{op.byte:02X}"] = 1.0 if op.byte == byte else 0.0
    if pin_fetched_byte:
        pins["fetched_opcode"] = float(byte)
    return pins


# State-input persist -> previous step's output, for fold_sequence over the
# 6502 graph.  pc and the trace-context bytes are pinned per step instead
# (sequence_pins knows them); ea feeds back as a byte split, so it must fold
# constant -- true for the write-free poll/spin windows this is built for.
SEQUENCE_FEEDBACK = {
    "current_a": "new_a", "current_x": "new_x", "current_y": "new_y",
    "current_s": "new_s", "current_halted": "new_halted",
    "current_p_c": "new_p_c_final", "current_p_z": "new_p_z_final",
    "current_p_i": "new_p_i_final", "current_p_d": "new_p_d_final",
    "current_p_b": "new_p_b_final", "current_p_u": 1.0,  # U reads 1 always
    "current_p_v": "new_p_v_final", "current_p_n": "new_p_n_final",
    "current_ir": "fetched_opcode",
    "current_oplo": "fetched_operand_lo",
    "current_ophi": "fetched_operand_hi",
    "current_ea_lo": ("ea", "lo"), "current_ea_hi": ("ea", "hi"),
    "current_data": "data_out",
}


def sequence_shape_pins(code: bytes, opcodes) -> tuple[list[dict[str, float]], list[int]]:
    """Per-step pins for a canonical window SHAPE: opcode one-hots and the
    fetched opcode byte only -- pc and operand bytes stay symbolic, so one
    fold serves every (head, operands) instantiation.  The router verifies
    the opcode byte at each returned offset; operand bytes need no check
    (the expert reads them through the live memory structure).  Returns
    (pin_list, pc_offsets) for fold_sequence(..., pc_offsets=...)."""
    pin_list: list[dict[str, float]] = []
    offsets: list[int] = []
    off = 0
    while off < len(code):
        op = opcodes.get(code[off])
        if op is None:
            raise ValueError(f"byte {code[off]:02X} at +{off} is not an opcode")
        pins = opcode_pins(op.byte, opcodes)
        pins["current_halted"] = 0.0
        pin_list.append(pins)
        offsets.append(off)
        off += op.length
    if off != len(code):
        raise ValueError("window does not end on an instruction boundary")
    return pin_list, offsets


def sequence_pins(head_pc: int, code: bytes, opcodes) -> list[dict[str, float]]:
    """Per-step pins for a K-instruction window: the bytes at head_pc.

    Each step gets its opcode pins plus its pc and every in-window operand
    byte pinned (the router verifies pc == head_pc and the whole window's
    bytes, so all of it is router-guaranteed).  current_halted is pinned 0:
    the run loop never dispatches a halted machine, and the runner checks
    each step's folded new_halted is constant 0.
    """
    pin_list = []
    pc = head_pc
    off = 0
    while off < len(code):
        op = opcodes.get(code[off])
        if op is None:
            raise ValueError(f"byte {code[off]:02X} at +{off} is not an opcode")
        pins = opcode_pins(op.byte, opcodes)
        pins["current_pc_lo"] = float(pc & 0xFF)
        pins["current_pc_hi"] = float((pc >> 8) & 0xFF)
        pins["current_halted"] = 0.0
        if off + 1 < len(code):
            pins["fetched_operand_lo_all"] = float(code[off + 1])
        if off + 2 < len(code):
            pins["fetched_operand_hi_all"] = float(code[off + 2])
        pin_list.append(pins)
        pc += op.length
        off += op.length
    if off != len(code):
        raise ValueError("window does not end on an instruction boundary")
    return pin_list
