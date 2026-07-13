"""Partial evaluation of a ProgramGraph into a rebuilt (folded) graph.

The opcode-routed engine (model/engine_gen.py) currently prunes the *union*
model in place: per expert opcode it keeps only the live subset of the
union's neurons and heads, but a ReGLU whose gate folds to a positive
constant under the pin (the taken branch of a mux) must still run as a
neuron, because the union arithmetic is shared.  That is the measured gap
between the in-place experts (~0.29x of the union terms kept) and full
partial evaluation (~0.195x): the missing factor is exactly the
gate=1 multiplies, which fold to plain linear maps and disappear into their
consumers' expressions.

``GraphFolder.fold(pins)`` produces a new ProgramGraph --
suitable for asap_schedule + build_model -- in which:

- pinned decode gates (``is_op_XX``/``is_nmi`` persists) are constants;
- a ReGLU whose folded gate is a non-positive constant (or whose folded
  value is the zero expression) is the zero function and is removed;
- a ReGLU whose folded gate is a *positive* constant is an exact linear
  map, inlined into every consumer (the rebuilt-expert lever);
- a persist whose folded expression is constant is inlined; a non-constant
  persist is kept (it still groups its expression into one slot);
- backward liveness from the folded output tokens plus ``root_names`` keeps
  only ops that still reach a readout through non-zero contributions.

Every surviving op keeps its union name.  LookUps carry provenance
(``union_lookup_of``, plus ``memory_group`` tags copied over) so an engine
can share attention state -- the 16x4KiB memory structure and the
running-argmax registers -- across expert models built from different folds
of the same union graph.  The shared-state contract is checked here, not
assumed: a lookup whose *insert-side* expressions (key/clear/values) can
reach a pinned gate in the union graph would insert pin-dependent history
into shared state, so it is reported in ``FoldResult.insert_side_hazards``
for the caller to refuse or special-case.

Folded-away root dims (a readout that folds to a passthrough or a
constant under the pin, e.g. ``new_y`` for an opcode that never touches Y)
are re-materialized as persists with the root's name, so a step_spec can
always resolve its readout slots by name.

NOTE: materialization calls ``reset_graph()`` -- the module-level graph
context afterwards holds the *folded* graph.  The union graph stays alive
through the ProgramGraph snapshot the folder was constructed with; any
later graph construction must call ``reset_graph()`` itself (build() does).
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
    PersistDimension,
    ProgramGraph,
    ReGLUDimension,
    reset_graph,
)


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


def graph_delta(start_d: int, start_l: int) -> tuple[int, int, int, int]:
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
    return nr, np_, nl, nt


def masked_exact_read(addr, byte_value, write_addr, is_write) -> Expression:
    """A memory_at site rebuilt flat, in the folded graph's context: ONE
    64K-span lookup (memory_group FUSED) plus the exact-address mask.
    Identical output to the union's 16-group sum -- the mask keeps only
    exact hits, and an exact hit answers from its own group either way --
    at ~8 dims instead of ~300 per read.  Expert folds only: the union's
    group split serves the exported artifact's f32 score range."""
    value, found = graph.fetch([byte_value, write_addr], query=addr,
                               key=write_addr, clear_key=1 - is_write,
                               tie_break="latest")
    value.lookup.memory_group = graph.MEMORY_GROUP_FUSED
    diff = addr - Expression({found: 1})
    abs_diff = graph.persist(graph.reglu(diff, diff) + graph.reglu(-diff, -diff))
    exact = graph.stepglu(Expression({graph.one: 1}), -abs_diff)
    return graph.make_multiply(Expression({value: 1}), exact)


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


class GraphFolder:
    """Folds one union ProgramGraph under different pin assignments.

    Build once per union graph; ``fold()`` per expert.  ``root_names`` are
    dim names that must exist in every folded graph (the runtime readout
    contract -- pass ``RUNTIME_READ_DIMS`` for the 6502 machine).
    """

    def __init__(self, program_graph: ProgramGraph, root_names=()):
        self.pg = program_graph
        self.root_names = tuple(root_names)
        self.by_name: dict[str, Dimension] = {}
        for d in program_graph.all_dims:
            self.by_name.setdefault(d.name, d)
        for name in self.root_names:
            if name not in self.by_name:
                raise KeyError(f"root dim {name!r} not in the graph")
        self.hazard_cache: dict[frozenset[str], set] = {}

    # ── Partial evaluation ───────────────────────────────────────────

    def partial_eval(self, pins: dict[str, float], frozen: frozenset = frozenset()):
        pg = self.pg
        one = pg.one
        val: dict = {one: Expression({one: 1.0})}
        folded: dict = {}
        for d in pg.all_dims:
            if isinstance(d, PersistDimension) and d.name in frozen:
                # Opaque leaf: bound to another step's output at
                # materialization (fold_sequence), never folded through.
                continue
            if isinstance(d, PersistDimension) and d.name in pins:
                val[d] = Expression({one: float(pins[d.name])})
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
                e = subst(d.expr, val)
                c = const_of(e, one)
                if c is not None:
                    val[d] = Expression({one: c})
                else:
                    folded[d] = (e,)
        lookup_exprs: dict = {}
        for lu in pg.all_lookups:
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

    def fold(self, pins: dict[str, float], label: str = "") -> FoldResult:
        pg = self.pg
        pins = dict(pins)
        val, folded, lookup_exprs = self.partial_eval(pins)
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

        live: set = set()
        stack = [d for d in seeds if d not in val and not isinstance(d, InputDimension)]
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
                if site is not None:
                    # Fused read: only the site's raw ingredients stay
                    # live; the 16-group fabric behind the persist dies.
                    for e in site.values():
                        deps |= set(subst(e, val).terms)
                else:
                    (e,) = folded[d]
                    deps = set(e.terms)
            elif isinstance(d, LookUpDimension):
                fx = lookup_exprs[d.lookup]
                for e in fx["query"] + fx["key"] + fx["values"]:
                    deps |= set(e.terms)
                for sib in d.lookup.dims:  # a live lookup produces all values
                    if sib not in live and sib not in val:
                        live.add(sib)
            for dep in deps:
                if dep not in val and not isinstance(dep, InputDimension) and dep not in live:
                    stack.append(dep)

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

        union_lookup_of: dict[int, int] = {}
        lookup_of_union: dict[int, LookUp] = {}
        insert_side_hazards: list[str] = []
        made = set()
        n_reglus = n_persists = n_lookups = n_terms = 0
        for d in pg.all_dims:
            if isinstance(d, InputDimension) or d in val or d not in live:
                continue
            if isinstance(d, ReGLUDimension):
                a, b = folded[d]
                nd = ReGLUDimension(tr(a), tr(b), name=d.name)
                sub[d] = Expression({nd: 1})
                n_reglus += 1
                n_terms += len(nd.a_expr.terms) + len(nd.b_expr.terms)
            elif isinstance(d, PersistDimension):
                site = getattr(d, "memory_read", None)
                if site is not None:
                    marks = len(graph.graph_dims), len(graph.graph_lookups)
                    expr = masked_exact_read(
                        tr(site["address"]), tr(site["byte_value"]),
                        tr(site["write_addr"]), tr(site["is_write"]),
                    )
                    nd = PersistDimension(expr, name=d.name)
                    sub[d] = Expression({nd: 1})
                    dr, dp, dl, dt = graph_delta(*marks)
                    n_reglus += dr
                    n_persists += dp
                    n_lookups += dl
                    n_terms += dt
                    continue
                (e,) = folded[d]
                nd = PersistDimension(tr(e), name=d.name)
                sub[d] = Expression({nd: 1})
                n_persists += 1
                n_terms += len(nd.expr.terms)
            elif isinstance(d, LookUpDimension):
                lu = d.lookup
                if lu in made:
                    continue
                made.add(lu)
                fx = lookup_exprs[lu]
                nlu = LookUp(
                    [tr(e) for e in fx["values"]],
                    [tr(e) for e in fx["query"]],
                    [tr(e) for e in fx["key"]],
                    tie_break=lu.tie_break,
                )
                nlu.name = lu.name
                mg = getattr(lu, "memory_group", None)
                if mg is not None:
                    nlu.memory_group = mg
                union_lookup_of[nlu.id] = lu.id
                lookup_of_union[lu.id] = nlu
                for ud, ndim in zip(lu.dims, nlu.dims, strict=True):
                    ndim.name = ud.name
                    sub[ud] = Expression({ndim: 1})
                if lu in hazards:
                    insert_side_hazards.append(lu.name or f"lookup_{lu.id}")
                n_lookups += 1
                for e in nlu.query_exprs_2d + nlu.key_exprs_2d + nlu.value_exprs:
                    n_terms += len(e.terms)

        # Folded-away roots come back as named persists so readout slots
        # always resolve by name (and stay alive via the output tokens).
        folded_roots: list[str] = []
        for d in root_dims:
            if d in val:
                nd = PersistDimension(tr(val[d]), name=d.name)
                sub[d] = Expression({nd: 1})
                folded_roots.append(d.name)
                n_persists += 1
                n_terms += len(nd.expr.terms)

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
            union_lookup_of=union_lookup_of,
            lookup_of_union=lookup_of_union,
            insert_side_hazards=tuple(insert_side_hazards),
            n_reglus=n_reglus,
            n_persists=n_persists,
            n_lookups=n_lookups,
            n_terms=n_terms,
            folded_roots=tuple(folded_roots),
        )


    # ── Sequence fold: K chained steps as one graph ───────────────────

    def fold_sequence(self, pin_list, feedback, label: str = "",
                      keep_alive_token: str = "READOUT",
                      pc_offsets=None) -> FoldResult:
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
        """
        pg = self.pg
        one = pg.one
        n_steps = len(pin_list)
        root_dims = [self.by_name[n] for n in self.root_names]
        feedback = dict(feedback)
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
        union_lookup_of: dict[int, int] = {}
        lookup_of_union: dict[int, LookUp] = {}
        insert_side_hazards: set[str] = set()
        folded_roots: list[str] = []
        keep_alive = Expression()
        n_reglus = n_persists = n_lookups = n_terms = 0
        fb_expr: dict[str, Expression] = {}  # prev step's outputs, new-graph space
        fb_const: dict[str, float] = {}
        new_outputs = None

        pc_names = ("current_pc_lo", "current_pc_hi")
        pc_exprs = None  # step-0 (head) pc byte expressions, new-graph space

        for k in range(n_steps):
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
            for name, v in pins.items():
                all_pins[f"s{k}:{name}"] = v

            val, folded, lookup_exprs = self.partial_eval(pins, frozenset(frozen))
            insert_side_hazards |= {
                lu.name or f"lookup_{lu.id}"
                for lu in self.insert_hazards(frozenset(pins))
            }

            # Liveness seeds: this step's roots, the outputs the next step
            # feeds on, and (final step) the graph's output tokens.
            fb_dims = [
                self.by_name[o[0] if isinstance(o, tuple) else o]
                for o in feedback.values() if not isinstance(o, (int, float))
            ] if k + 1 < n_steps else []
            fout = {}
            if k + 1 == n_steps:
                val_out = {d: v for d, v in val.items() if d not in set(root_dims)}
                fout = {
                    name: subst(expr, val_out)
                    for name, expr in pg.output_tokens.items()
                    if isinstance(expr, Expression)
                }
            seeds: set = set()
            for e in fout.values():
                seeds |= set(e.terms)
            for d in root_dims + fb_dims:
                seeds |= set(val[d].terms) if d in val else {d}

            def is_leaf(d, *, frozen=frozen):
                return (
                    isinstance(d, InputDimension)
                    or (isinstance(d, PersistDimension) and d.name in frozen)
                )

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
                    if site is not None:
                        # Fused read: only the site's raw ingredients stay
                        # live; the 16-group fabric behind the persist dies.
                        for e in site.values():
                            deps |= set(subst(e, val).terms)
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
                for dep in deps:
                    if dep not in val and not is_leaf(dep) and dep not in live:
                        stack.append(dep)

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
                from transformer_nes.graph.core import stepglu

                off = int(pc_offsets[k])
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

            made = set()
            for d in pg.all_dims:
                if d in val or d not in live or is_leaf(d):
                    continue
                if isinstance(d, ReGLUDimension):
                    a, b = folded[d]
                    nd = ReGLUDimension(tr(a), tr(b), name=sname(d.name))
                    sub[d] = Expression({nd: 1})
                    n_reglus += 1
                    n_terms += len(nd.a_expr.terms) + len(nd.b_expr.terms)
                elif isinstance(d, PersistDimension):
                    site = getattr(d, "memory_read", None)
                    if site is not None:
                        marks = len(graph.graph_dims), len(graph.graph_lookups)
                        expr = masked_exact_read(
                            tr(site["address"]), tr(site["byte_value"]),
                            tr(site["write_addr"]), tr(site["is_write"]),
                        )
                        nd = PersistDimension(expr, name=sname(d.name))
                        sub[d] = Expression({nd: 1})
                        dr, dp, dl, dt = graph_delta(*marks)
                        n_reglus += dr
                        n_persists += dp
                        n_lookups += dl
                        n_terms += dt
                        continue
                    (e,) = folded[d]
                    nd = PersistDimension(tr(e), name=sname(d.name))
                    sub[d] = Expression({nd: 1})
                    n_persists += 1
                    n_terms += len(nd.expr.terms)
                elif isinstance(d, LookUpDimension):
                    lu = d.lookup
                    if lu in made:
                        continue
                    made.add(lu)
                    fx = lookup_exprs[lu]
                    nlu = LookUp(
                        [tr(e) for e in fx["values"]],
                        [tr(e) for e in fx["query"]],
                        [tr(e) for e in fx["key"]],
                        tie_break=lu.tie_break,
                    )
                    nlu.name = sname(lu.name)
                    mg = getattr(lu, "memory_group", None)
                    if mg is not None:
                        nlu.memory_group = mg
                    union_lookup_of[nlu.id] = lu.id
                    lookup_of_union[lu.id] = nlu  # last step wins; ids share state
                    for ud, ndim in zip(lu.dims, nlu.dims, strict=True):
                        ndim.name = sname(ud.name)
                        sub[ud] = Expression({ndim: 1})
                    n_lookups += 1
                    for e in nlu.query_exprs_2d + nlu.key_exprs_2d + nlu.value_exprs:
                        n_terms += len(e.terms)

            # Roots: every step's readout set, suffixed; folded-away roots
            # come back as named persists (same contract as fold()).
            for d in root_dims:
                if d in val:
                    nd = PersistDimension(tr(val[d]), name=sname(d.name))
                    sub[d] = Expression({nd: 1})
                    folded_roots.append(nd.name)
                    n_persists += 1
                    n_terms += len(nd.expr.terms)
                keep_alive = keep_alive + 1e-6 * sub[d]

            if pc_offsets is not None and k == 0:
                pc_exprs = tuple(
                    tr(Expression({self.by_name[n]: 1})) for n in pc_names
                )

            fb_const, fb_expr = {}, {}
            for out in set(feedback.values()):
                if isinstance(out, (int, float)):
                    continue
                out_name = out[0] if isinstance(out, tuple) else out
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
                new_outputs = {
                    name: tr(fout[name]) if name in fout else expr
                    for name, expr in pg.output_tokens.items()
                }

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
            union_lookup_of=union_lookup_of,
            lookup_of_union=lookup_of_union,
            insert_side_hazards=tuple(sorted(insert_side_hazards)),
            n_reglus=n_reglus,
            n_persists=n_persists,
            n_lookups=n_lookups,
            n_terms=n_terms,
            folded_roots=tuple(folded_roots),
            n_steps=n_steps,
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
