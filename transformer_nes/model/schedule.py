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
    MemoryRead,
    PersistDimension,
    ReGLUDimension,
)

logger = logging.getLogger(__name__)


ATTENTION_OPS = (LookUp, MemoryRead)


def build_graph(program_graph):
    """Build scheduling dependencies from one captured graph."""
    all_dims = program_graph.all_dims
    all_lookups = program_graph.all_lookups
    all_memory_reads = program_graph.all_memory_reads
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
            for expr in lu.query_exprs + lu.key_exprs + lu.value_exprs:
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

    return dict(
        ops=ops,
        inputs=inputs,
        produced=produced,
        dim_to_op=dim_to_op,
        op_deps=op_deps,
        children=children,
        consumers=consumers,
    )


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

    return num_layers, std_layers, alive_after


def legal_phases(lo, hi, op):
    """Phases of op's parity within [lo, hi]."""
    if isinstance(op, ATTENTION_OPS):
        start, step = lo + (-lo) % 4, 4
    elif isinstance(op, ReGLUDimension):
        start, step = lo + (2 - lo % 4 + 4) % 4, 4
    else:
        start, step = lo + (0 if lo % 2 == 1 else 1), 2
    return range(start, hi + 1, step)


def asap_schedule(program_graph, log=None):
    """Load-balanced critical-path schedule for a captured graph.

    Layer count equals the dependency critical path. A reverse-topological
    pass places every operation at the least-loaded legal phase within its
    dependency window, breaking ties toward later phases to shorten lifetimes.
    """
    log = log or logger.debug
    graph = build_graph(program_graph)

    ops = graph["ops"]
    op_deps = graph["op_deps"]
    children = graph["children"]
    all_dims = all_result_dims(graph)

    # Weight construction adds constant-one gates for linear passthroughs
    # and erases after scheduling. Their implicit input must never be recycled.
    output_dims = {program_graph.one}
    for expr in program_graph.output_tokens.values():
        if isinstance(expr, Expression):
            output_dims |= set(expr.terms.keys())

    if not ops:
        log("ASAP schedule: 0 ops, 0 layers, max alive width 0, busiest layer 0 ops")
        return dict(std_layers=[], num_layers=0, alive_after={})

    # ASAP with phase parity. Iterate in
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

    # Reverse-topological leveling: children are already placed, so
    # each op's window is [its ASAP phase, min(children) - 1].  ASAP
    # phases of parents are strictly below the op's, so windows stay
    # nonempty as parents are placed after their children.
    max_phase = max(asap.values())
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

    (num_layers, std_layers, alive_after) = layers_and_liveness(
        pa, all_dims, graph["dim_to_op"], graph["consumers"], output_dims
    )
    max_dep = max((len(s) for s in alive_after.values()), default=0)
    busiest = max((len(p1) + len(f) + len(p2) for _, p1, f, p2 in std_layers), default=0)
    log(
        f"Balanced schedule: {len(ops)} ops, "
        f"{num_layers} layers, max alive width {max_dep}, busiest layer {busiest} ops"
    )
    return dict(
        std_layers=std_layers,
        num_layers=num_layers,
        alive_after=alive_after,
    )
