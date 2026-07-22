"""Schedulers: ops onto 4-phase transformer layers.

Assigns each gate (LookUp/MemoryRead, ReGLU, Persist) to a 4-phase layer:
  phase 0: Attention or external read
  phase 1: Persist1
  phase 2: FFN (ReGLU)
  phase 3: Persist2

Two entry points with one output shape (consumed by model/weights.py):
``asap_schedule`` (critical path + load-balanced leveling; the production
scheduler at every scale) and ``milp_schedule`` (the opt-in per-expert
certificate: exact total-lifetime objective under a width cap).
"""

import logging
from collections import defaultdict

from pulp import (
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    LookUp,
    LookUpDimension,
    MemoryRead,
    MemoryReadDimension,
    PersistDimension,
    ReGLUDimension,
    graph_dims,
    graph_lookups,
    graph_memory_reads,
)

logger = logging.getLogger(__name__)


ATTENTION_OPS = (LookUp, MemoryRead)
ATTENTION_DIMS = (LookUpDimension, MemoryReadDimension)


def build_graph(all_dims=None, all_lookups=None, all_memory_reads=None):
    """Build dependency graph from dim/lookup lists.

    Args:
        all_dims: list of Dimension objects (defaults to module-global graph_dims)
        all_lookups: list of LookUp objects (defaults to module-global graph_lookups)
    """
    explicit_graph = all_dims is not None or all_lookups is not None
    if all_dims is None:
        all_dims = graph_dims
    if all_lookups is None:
        all_lookups = graph_lookups
    if all_memory_reads is None:
        all_memory_reads = (
            list(dict.fromkeys(
                dim.memory_read
                for dim in all_dims
                if isinstance(dim, MemoryReadDimension)
            ))
            if explicit_graph else graph_memory_reads
        )
    inputs = [d for d in all_dims if isinstance(d, InputDimension)]
    reglus = [d for d in all_dims if isinstance(d, ReGLUDimension)]
    persists = [d for d in all_dims if isinstance(d, PersistDimension)]
    lookups = list(all_lookups) + list(all_memory_reads)
    ops = reglus + persists + lookups

    produced = {}
    for r in reglus:
        produced[r] = {r}
    for p in persists:
        produced[p] = {p}
    for lu in lookups:
        produced[lu] = set(lu.dims)

    def edeps(expr):
        return set(expr.terms.keys()) if isinstance(expr, Expression) else set()

    deps_cache = {}
    for r in reglus:
        deps_cache[r] = edeps(r.a_expr) | edeps(r.b_expr)
    for p in persists:
        deps_cache[p] = edeps(p.expr)
    for lu in lookups:
        d = set()
        if isinstance(lu, MemoryRead):
            d |= edeps(lu.address)
        else:
            for expr in lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs:
                d |= edeps(expr)
        deps_cache[lu] = d

    dim_to_op = {}
    for op in ops:
        for d in produced[op]:
            dim_to_op[d] = op

    op_deps = defaultdict(set)
    children = defaultdict(set)
    consumers = defaultdict(set)
    for op in ops:
        for dim in deps_cache[op]:
            consumers[dim].add(op)
            if dim in dim_to_op and dim_to_op[dim] != op:
                pred = dim_to_op[dim]
                op_deps[op].add(pred)
                children[pred].add(op)

    avg_lookups = {
        lu for lu in lookups
        if isinstance(lu, LookUp) and lu.tie_break == "average"
    }
    tight_to = defaultdict(set)
    for op in reglus + persists:
        for dim in deps_cache[op]:
            if isinstance(dim, ATTENTION_DIMS) and dim in dim_to_op:
                lu = dim_to_op[dim]
                if lu in avg_lookups:
                    tight_to[op].add(lu)

    return dict(
        ops=ops,
        reglus=reglus,
        persists=persists,
        lookups=lookups,
        inputs=inputs,
        produced=produced,
        deps_cache=deps_cache,
        dim_to_op=dim_to_op,
        op_deps=op_deps,
        children=children,
        consumers=consumers,
        tight_to=dict(tight_to),
    )


def min_layers(ops, op_deps):
    """Critical path length with phase parity (ASAP without tight constraints)."""
    phase, remaining = {}, set(ops)
    while remaining:
        progress = False
        for op in list(remaining):
            if not all(p in phase for p in op_deps[op]):
                continue
            lo = max((phase[p] for p in op_deps[op]), default=-1) + 1
            if isinstance(op, ATTENTION_OPS):
                lo += (-lo) % 4
            elif isinstance(op, ReGLUDimension):
                lo += (2 - lo % 4 + 4) % 4
            else:
                lo += 0 if lo % 2 == 1 else 1
            phase[op] = lo
            remaining.discard(op)
            progress = True
        assert progress, "Cycle in dependencies"
    return max(phase.values()) // 4 + 1


def all_result_dims(graph):
    """All dimensions in the graph (inputs + produced)."""
    dims = list(graph["inputs"])
    dim_set = set(dims)
    for op in graph["ops"]:
        for d in graph["produced"][op]:
            if d not in dim_set:
                dim_set.add(d)
                dims.append(d)
    return dims


def layers_and_liveness(pa, all_dims, dim_to_op, consumers, output_dims):
    """Turn a phase assignment into layers, birth/death, and alive sets.

    Shared by the MILP extraction and the ASAP scheduler; the phase parity
    convention is LookUp -> 4L, Persist -> 4L+1 or 4L+3, ReGLU -> 4L+2.
    """
    max_phase = max(pa.values())
    num_layers = max_phase // 4 + 1

    by_phase = defaultdict(list)
    for op, p in pa.items():
        by_phase[p].append(op)

    std_layers = []
    for L in range(num_layers):
        std_layers.append(
            (
                [op for op in by_phase.get(4 * L, []) if isinstance(op, ATTENTION_OPS)],
                [op for op in by_phase.get(4 * L + 1, []) if isinstance(op, PersistDimension)],
                [op for op in by_phase.get(4 * L + 2, []) if isinstance(op, ReGLUDimension)],
                [op for op in by_phase.get(4 * L + 3, []) if isinstance(op, PersistDimension)],
            )
        )

    dim_birth = {}
    for d in all_dims:
        if isinstance(d, InputDimension):
            dim_birth[d] = -1
        else:
            prod = dim_to_op.get(d)
            if prod and prod in pa:
                dim_birth[d] = pa[prod]

    dim_death = {}
    last_boundary = 4 * num_layers - 1
    for d in all_dims:
        if d in output_dims:
            dim_death[d] = last_boundary + 1
            continue
        last = -1
        for c_op in consumers.get(d, set()):
            if c_op in pa:
                last = max(last, pa[c_op])
        if last >= 0:
            dim_death[d] = last
        elif d in dim_birth:
            dim_death[d] = dim_birth[d]

    def alive_at(c):
        return frozenset(
            d
            for d in all_dims
            if d in dim_birth and d in dim_death and dim_birth[d] <= c and dim_death[d] > c
        )

    alive_after = {}
    for L in range(num_layers):
        for sp in (1, 3):
            c = 4 * L + sp
            alive_after[c] = alive_at(c)

    return num_layers, max_phase, std_layers, dim_birth, dim_death, alive_after


def legal_phases(lo, hi, op):
    """Phases of op's parity within [lo, hi]."""
    if isinstance(op, ATTENTION_OPS):
        start, step = lo + (-lo) % 4, 4
    elif isinstance(op, ReGLUDimension):
        start, step = lo + (2 - lo % 4 + 4) % 4, 4
    else:
        start, step = lo + (0 if lo % 2 == 1 else 1), 2
    return range(start, hi + 1, step)


def asap_schedule(input_tokens, output_tokens, log=None, program_graph=None,
                  balance=True):
    """Critical-path (ASAP) schedule, optionally load-balanced.

    Same output shape as milp_schedule but no solver: layer count equals
    the dependency critical path.  This is the practical path for large
    graphs (the union machine's MILP does not solve in useful time -- the
    blowup is the formulation's per-dim/per-boundary alive indicators,
    not the HiGHS engine).

    Plain ASAP piles ops into the earliest layers, and d_model ends up
    dominated by the busiest layer's head/FFN demand rather than by
    liveness.  With balance=True a reverse-topological leveling pass
    re-places every op at the least-loaded legal phase within its
    dependency window (ties -> latest, which also shortens lifetimes).
    """
    log = log or logger.info

    if program_graph is not None:
        pg = program_graph
        graph = build_graph(pg.all_dims, pg.all_lookups, pg.all_memory_reads)
    else:
        graph = build_graph()

    ops = graph["ops"]
    op_deps = graph["op_deps"]
    children = graph["children"]
    if graph.get("tight_to"):
        raise NotImplementedError(
            "asap_schedule does not honor same-layer constraints of "
            "average-tiebreak lookups; use milp_schedule"
        )
    all_dims = all_result_dims(graph)

    output_dims = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            output_dims |= set(expr.terms.keys())

    if not ops:
        log("ASAP schedule: 0 ops, 0 layers, max alive width 0, busiest layer 0 ops")
        return dict(
            phase_assign={}, std_layers=[], num_layers=0,
            dim_birth={}, dim_death={}, alive_after={}, width=None,
        )

    # ASAP with phase parity (the same rule min_layers uses).  Iterate in
    # stable ops order, not set order: the schedule (and therefore the
    # constructed weights) must be deterministic across processes.
    asap, remaining = {}, set(ops)
    topo_order = []
    while remaining:
        progress = False
        for op in ops:
            if op not in remaining or any(p not in asap for p in op_deps[op]):
                continue
            lo = max((asap[p] for p in op_deps[op]), default=-1) + 1
            asap[op] = next(iter(legal_phases(lo, lo + 3, op)))
            topo_order.append(op)
            remaining.discard(op)
            progress = True
        assert progress, "Cycle in dependencies"

    pa = dict(asap)
    if balance:
        # Reverse-topological leveling: children are already placed, so
        # each op's window is [its ASAP phase, min(children) - 1].  ASAP
        # phases of parents are strictly below the op's, so windows stay
        # nonempty as parents are placed after their children.
        max_phase = max(pa.values())
        pa = {}
        load = defaultdict(int)
        for op in reversed(topo_order):
            kids = [pa[c] for c in children.get(op, ()) if c in pa]
            hi = (min(kids) - 1) if kids else max_phase
            hi = max(hi, asap[op])
            candidates = list(legal_phases(asap[op], hi, op)) or [asap[op]]
            best = min(candidates, key=lambda p: (load[p], -p))
            pa[op] = best
            load[best] += 1

    (num_layers, max_phase, std_layers, dim_birth, dim_death, alive_after) = (
        layers_and_liveness(
            pa, all_dims, graph["dim_to_op"], graph["consumers"], output_dims
        )
    )
    max_dep = max((len(s) for s in alive_after.values()), default=0)
    busiest = max(
        (len(p1) + len(f) + len(p2) for _, p1, f, p2 in std_layers), default=0
    )
    log(
        f"{'Balanced' if balance else 'ASAP'} schedule: {len(ops)} ops, "
        f"{num_layers} layers, max alive width {max_dep}, busiest layer {busiest} ops"
    )
    return dict(
        phase_assign=pa,
        std_layers=std_layers,
        num_layers=num_layers,
        dim_birth=dim_birth,
        dim_death=dim_death,
        alive_after=alive_after,
        width=None,
    )


def milp_schedule(
    input_tokens, output_tokens, max_layers=None, log=None,
    program_graph=None, time_limit=3600, width_bound=None,
):
    """Optimal-lifetime schedule: the per-expert certificate.

    Minimizes total dim lifetime (every phase a value outlives its
    producer is one emitted passthrough/erase statement, attention or FFN
    side) plus the attention passthrough sum, under a width cap.

    Args:
        max_layers: maximum number of transformer layers.
        log: logging callable (defaults to logger.info).
        program_graph: optional ProgramGraph; if provided, uses its dims/lookups
            and positional dimensions instead of module-level globals.
        time_limit: solver budget in seconds; at the limit HiGHS returns its
            best incumbent (the schedule is feasible, just not proven optimal).
        width_bound: optional cap on d_model (2 * D_half <= width_bound),
            usually the balanced-ASAP d_model.
    """
    log = log or print

    if program_graph is not None:
        pg = program_graph
        graph = build_graph(pg.all_dims, pg.all_lookups, pg.all_memory_reads)
    else:
        graph = build_graph()
    ops = graph["ops"]
    od = graph["op_deps"]
    tt = graph.get("tight_to", {})
    consumers = graph["consumers"]
    dim_to_op = graph["dim_to_op"]

    all_dims = all_result_dims(graph)

    output_dims = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            output_dims |= set(expr.terms.keys())

    N = max_layers or min_layers(ops, od)
    P = 4 * N
    log(f"MILP: {len(ops)} ops, {len(all_dims)} dims, {N} layers, {P} phases")

    # ── MILP ──────────────────────────────────────────────────────
    prob = LpProblem("schedule", LpMinimize)
    D_half = LpVariable("D_half", 0, cat="Integer")

    k = {op: LpVariable(f"k_{i}", 0, N - 1, LpInteger) for i, op in enumerate(ops)}
    z = {
        op: LpVariable(f"z_{i}", 0, 1, LpBinary)
        for i, op in enumerate(ops)
        if isinstance(op, PersistDimension)
    }

    def phase_of(op):
        if isinstance(op, ATTENTION_OPS):
            return 4 * k[op]
        if isinstance(op, ReGLUDimension):
            return 4 * k[op] + 2
        return 4 * k[op] + 1 + 2 * z[op]

    for op in ops:
        for dep in od.get(op, set()):
            if dep in k:
                prob += phase_of(op) >= phase_of(dep) + 1

    for op, lus in tt.items():
        for lu in lus:
            if lu in k and op in k:
                prob += k[op] == k[lu]

    # death[d] = max phase of any consumer (LP variable)
    death = {}
    for d in all_dims:
        if d in output_dims:
            continue
        cons = [c_op for c_op in consumers.get(d, set()) if c_op in k]
        if not cons:
            continue
        dv = LpVariable(f"d_{id(d)}", 0, P - 1, LpInteger)
        for c_op in cons:
            prob += dv >= phase_of(c_op)
        death[d] = dv

    di = {d: i for i, d in enumerate(all_dims)}
    lookups = graph["lookups"]
    persists = graph["persists"]
    deps_cache = graph["deps_cache"]

    # ── Layer indicators for lookups ─────────────────────────────
    lu_at = {}
    for lu in lookups:
        for L in range(N):
            lu_at[lu, L] = LpVariable(f"la_{id(lu)}_{L}", 0, 1, LpBinary)
        prob += lpSum(lu_at[lu, L] for L in range(N)) == 1
        prob += k[lu] == lpSum(L * lu_at[lu, L] for L in range(N))

    # ── Layer indicators for persists (persist1 vs persist2) ─────
    p_layer = {}
    p1_at = {}
    for p in persists:
        for L in range(N):
            p_layer[p, L] = LpVariable(f"pl_{id(p)}_{L}", 0, 1, LpBinary)
        prob += lpSum(p_layer[p, L] for L in range(N)) == 1
        prob += k[p] == lpSum(L * p_layer[p, L] for L in range(N))
        for L in range(N):
            p1v = LpVariable(f"p1_{id(p)}_{L}", 0, 1, LpBinary)
            prob += p1v <= p_layer[p, L]
            prob += p1v <= 1 - z[p]
            prob += p1v >= p_layer[p, L] + (1 - z[p]) - 1
            p1_at[p, L] = p1v

    # ── Passthrough indicators for attention ─────────────────────
    # For each persist1 dep d, pd[d,L]=1 iff d needs passthrough at layer L
    P1_deps = defaultdict(set)
    for p in persists:
        for d in deps_cache[p]:
            if isinstance(d, (InputDimension, *ATTENTION_DIMS, ReGLUDimension, PersistDimension)):
                P1_deps[d].add(p)

    pd_var = {}
    for d, p_set in P1_deps.items():
        lu_of_d = dim_to_op.get(d) if isinstance(d, ATTENTION_DIMS) else None
        for L in range(N):
            v = LpVariable(f"pd_{id(d)}_{L}", 0, 1, LpBinary)
            prob += v <= lpSum(p1_at[p, L] for p in p_set)
            if lu_of_d is not None and lu_of_d in lu_at:
                prob += v <= 1 - lu_at[lu_of_d, L]
                for p in p_set:
                    prob += v >= p1_at[p, L] - lu_at[lu_of_d, L]
            else:
                for p in p_set:
                    prob += v >= p1_at[p, L]
            pd_var[d, L] = v

    # ── Lookup heads and dims ───────────────────────────────────
    lu_h = {lu: (len(lu.value_exprs) + 1) // 2 for lu in lookups}
    lu_d = {lu: len(lu.dims) for lu in lookups}

    # ── needs_slot indicator for lookup/reglu dims ─────────────
    # A lookup dim at phase 4L needs a slot only if death >= 4L+2
    # (i.e., it survives past the persist1 boundary at 4L+1).
    # Similarly for reglu dims at phase 4L+2: needs slot if death >= 4L+4.
    # Persist dims always need slots.
    ns = {}
    for d in all_dims:
        if d in output_dims:
            continue
        if d not in death:
            continue
        prod = dim_to_op.get(d)
        if prod is None or prod not in k:
            continue
        if isinstance(d, (*ATTENTION_DIMS, ReGLUDimension)):
            ns_v = LpVariable(f"ns_{di[d]}", 0, 1, LpBinary)
            prob += death[d] >= phase_of(prod) + 2 - P * (1 - ns_v)
            prob += death[d] <= phase_of(prod) + 1 + P * ns_v
            ns[d] = ns_v

    # ── Occupied slot count at each boundary (death >= c-1) ──────
    # For dims that need a slot: occupied(c) = |{d : needs_slot AND
    #   birth <= c AND death >= c-1}|.
    # Internal dims (lookup/reglu consumed within same half-layer)
    # don't occupy slots and are excluded via needs_slot.
    #
    # Also compute alive(c) = |{d : birth <= c AND death > c}|
    # for the head count dying computation.
    alive_sum = {}
    n_inputs = sum(1 for d in all_dims if isinstance(d, InputDimension))

    for c in range(P):
        if c % 2 == 0:
            continue
        ew = []
        alive = []
        for d in all_dims:
            prod = dim_to_op.get(d)
            is_input = isinstance(d, InputDimension)

            if d in output_dims:
                if is_input:
                    ew.append(1)
                    alive.append(1)
                elif prod in k:
                    bb = LpVariable(f"b_{di[d]}_{c}", 0, 1, LpBinary)
                    prob += phase_of(prod) <= c + P * (1 - bb)
                    prob += phase_of(prod) >= (c + 1) - P * bb
                    ew.append(bb)
                    alive.append(bb)
                continue

            if d not in death:
                continue

            if is_input:
                eu = LpVariable(f"ew_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c - 1) - P * (1 - eu)
                prob += death[d] <= (c - 2) + P * eu
                ew.append(eu)
                au = LpVariable(f"a_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c + 1) - P * (1 - au)
                prob += death[d] <= c + P * au
                alive.append(au)
            elif prod in k:
                bb = LpVariable(f"b_{di[d]}_{c}", 0, 1, LpBinary)
                prob += phase_of(prod) <= c + P * (1 - bb)
                prob += phase_of(prod) >= (c + 1) - P * bb
                eu = LpVariable(f"eu_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c - 1) - P * (1 - eu)
                prob += death[d] <= (c - 2) + P * eu
                ev = LpVariable(f"ew_{di[d]}_{c}", 0, 1, LpBinary)
                if d in ns:
                    prob += ev <= bb
                    prob += ev <= eu
                    prob += ev <= ns[d]
                    prob += ev >= bb + eu + ns[d] - 2
                else:
                    prob += ev <= bb
                    prob += ev <= eu
                    prob += ev >= bb + eu - 1
                ew.append(ev)
                au = LpVariable(f"au_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c + 1) - P * (1 - au)
                prob += death[d] <= c + P * au
                av = LpVariable(f"a_{di[d]}_{c}", 0, 1, LpBinary)
                prob += av <= bb
                prob += av <= au
                prob += av >= bb + au - 1
                alive.append(av)

        prob += 2 * D_half >= lpSum(ew)
        alive_sum[c] = lpSum(alive)

    # ── Head count constraint at each layer ──────────────────────
    # heads_L = n_lu + ceil((dying + passthrough) / 2)
    # dying_L = alive_prev - alive_cur + born_L
    for L in range(N):
        c_attn = 4 * L + 1
        c_prev = 4 * L - 1

        n_lu_L = lpSum(lu_h[lu] * lu_at[lu, L] for lu in lookups)
        pt_L = lpSum(pd_var[d, L] for d in P1_deps if (d, L) in pd_var)

        born_L = lpSum(lu_d[lu] * lu_at[lu, L] for lu in lookups) + lpSum(
            p1_at[p, L] for p in persists
        )
        prev_alive = alive_sum.get(c_prev, n_inputs)
        cur_alive = alive_sum[c_attn]
        dying_L = prev_alive - cur_alive + born_L

        prob += 2 * D_half >= 2 * n_lu_L + dying_L + pt_L

    # ── Objective / width bound ───────────────────────────────────
    if width_bound is not None:
        prob += 2 * D_half <= int(width_bound)
    # Total lifetime prices every carried slot -- attention and FFN --
    # linearly in variables the formulation already has; the attention
    # passthrough sum and a small width term break ties.
    births = lpSum(
        phase_of(dim_to_op[d]) for d in death
        if dim_to_op.get(d) in k
    )
    prob.setObjective(
        lpSum(death.values()) - births
        + lpSum(pd_var.values()) + 0.001 * D_half
    )

    # ── Solve ─────────────────────────────────────────────────────
    log("Solving MILP...")
    try:
        from pulp import HiGHS

        solver = HiGHS(timeLimit=time_limit)
    except Exception:
        from pulp import PULP_CBC_CMD

        solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    if prob.status != 1:
        raise RuntimeError(f"MILP infeasible (status={prob.status}); try more layers")

    lifetime = int(round(value(prob.objective)))
    opt_D = 2 * int(round(value(D_half)))

    # ── Extract assignment & build layers ──────────────────────────
    pa = {}
    for op in ops:
        layer = int(round(value(k[op])))
        if isinstance(op, ATTENTION_OPS):
            pa[op] = 4 * layer
        elif isinstance(op, ReGLUDimension):
            pa[op] = 4 * layer + 2
        else:
            is_p2 = int(round(value(z[op])))
            pa[op] = 4 * layer + 1 + 2 * is_p2

    (num_layers, _max_phase, std_layers, dim_birth, dim_death, alive_after) = (
        layers_and_liveness(pa, all_dims, dim_to_op, consumers, output_dims)
    )
    log(f"MILP schedule: {num_layers} layers, d_model {opt_D}, "
        f"objective {lifetime}")

    return dict(
        phase_assign=pa,
        std_layers=std_layers,
        num_layers=num_layers,
        dim_birth=dim_birth,
        dim_death=dim_death,
        alive_after=alive_after,
        width=opt_D,
    )
