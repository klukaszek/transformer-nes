"""Scheduler: ops onto 4-phase transformer layers.

Assigns each gate (LookUp/MemoryRead, ReGLU, Persist) to a 4-phase layer:
  phase 0: Attention or external read
  phase 1: Persist1
  phase 2: FFN (ReGLU)
  phase 3: Persist2

``asap_schedule`` (critical path + load-balanced leveling) is the sole
entry point; its output is consumed by model/weights.py.
"""

import logging
from collections import defaultdict

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

    The phase parity convention is LookUp -> 4L, Persist -> 4L+1 or 4L+3,
    ReGLU -> 4L+2.
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

    Layer count equals the dependency critical path, which is also the
    floor any scheduler can reach.

    Plain ASAP piles ops into the earliest layers, and d_model ends up
    dominated by the busiest layer's head/FFN demand rather than by
    liveness.  With balance=True a reverse-topological leveling pass
    re-places every op at the least-loaded legal phase within its
    dependency window (ties -> latest, which also shortens lifetimes).
    """
    log = log or logger.debug

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
            "average-tiebreak lookups need same-layer constraints that no "
            "scheduler here honors; every lookup must declare LATEST"
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

