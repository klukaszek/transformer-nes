"""MoE scouting probe: what does opcode-routed conditional computation buy?

Method: build the 6502 graph (bcd=False, the NES variant), then for each
candidate expert (single opcode / operation family / NMI) *partially
evaluate* the graph with the decode one-hot gates pinned to constants --
exactly the information a hard router would supply.  Constant-folding
propagates through ReGLU/persist/mux structure with linear inlining for
known-gate multiplies (so mux(gate=1, a, b) folds to `a` and the dead
branch is prunable).  Backward liveness from the runtime readout dims
then gives the expert's surviving subgraph, and the scheduler's own
phase-parity critical path (_min_layers) gives its layer depth.

Metrics per expert:
  - live ops (reglu / persist / lookup)
  - folded term count (proxy for CSR nnz = the load-bound cost)
  - layer depth via _min_layers on the pruned dependency graph

Baseline = same analysis with no pins; it reproduces the union graph
exactly (71 layers), so ratios are apples-to-apples.

Run: uv run python experiments/moe_probe/moe_probe.py
Results land next to this script as moe_probe_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from transformer_vm.graph import core as g
from transformer_vm.graph.core import (
    Expression,
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    reset_graph,
)
from transformer_vm.cpu6502.interpreter import build
from transformer_vm.cpu6502.spec import OPCODES
from transformer_vm.cpu6502.evaluator import _RUNTIME_READ_DIMS
from transformer_vm.scheduler.milp import _build_graph, _min_layers

OUT_DIR = Path(__file__).resolve().parent

# ── Build the graph once ────────────────────────────────────────────

reset_graph()
t0 = time.time()
input_tokens, output_tokens = build(bcd=False)
ALL_DIMS = list(g._all_dims)
ALL_LOOKUPS = list(g._all_lookups)
ONE = g.one
print(
    f"graph built in {time.time() - t0:.1f}s: {len(ALL_DIMS)} dims, "
    f"{len(ALL_LOOKUPS)} lookups",
    file=sys.stderr,
)

NAME2DIM: dict[str, object] = {}
for d in ALL_DIMS:
    if d.name not in NAME2DIM:
        NAME2DIM[d.name] = d

OP_GATE_NAMES = [f"is_op_{op.byte:02X}" for op in OPCODES.values()]
for n in OP_GATE_NAMES + ["is_nmi"]:
    assert n in NAME2DIM, f"missing gate dim {n}"

ROOT_DIMS = [NAME2DIM[n] for n in _RUNTIME_READ_DIMS]
for expr in output_tokens.values():
    if isinstance(expr, Expression):
        ROOT_DIMS.extend(expr.terms.keys())


# ── Partial evaluation ──────────────────────────────────────────────
# val[dim] is an Expression over "base" dims (opaque dims + `one`).
# A dim absent from val is its own base.  Constants live on `one`.


def _const_of(e: Expression):
    """Return the float value if e is constant (only `one` terms), else None."""
    for d in e.terms:
        if d is not ONE:
            return None
    return e.terms.get(ONE, 0.0)


def _subst(expr: Expression, val: dict) -> Expression:
    out = Expression()
    for d, c in expr.terms.items():
        v = val.get(d)
        if v is None:
            out = out + Expression({d: c})
        else:
            out = out + v * c
    return out


def analyze(pins: dict[str, float], label: str) -> dict:
    """Partial-eval + liveness + depth for one pin assignment."""
    val: dict = {ONE: Expression({ONE: 1.0})}
    # Folded expression per op-dim, used for dep edges and term counts.
    folded: dict = {}

    for d in ALL_DIMS:
        if d.name in pins and isinstance(d, PersistDimension):
            val[d] = Expression({ONE: float(pins[d.name])})
            continue
        if isinstance(d, ReGLUDimension):
            a = _subst(d.a_expr, val)
            b = _subst(d.b_expr, val)
            bc = _const_of(b)
            ac = _const_of(a)
            if bc is not None and bc <= 0:
                val[d] = Expression()
            elif ac is not None and ac == 0:
                val[d] = Expression()
            elif bc is not None and ac is not None:
                val[d] = Expression({ONE: ac * bc})
            elif bc is not None:
                # relu(b)*a with b a known positive constant: exact linear map.
                val[d] = a * bc
            else:
                folded[d] = (a, b)
        elif isinstance(d, PersistDimension):
            e = _subst(d.expr, val)
            c = _const_of(e)
            if c is not None:
                val[d] = Expression({ONE: c})
            else:
                folded[d] = (e,)
        elif isinstance(d, LookUpDimension):
            pass  # opaque; handled per-lookup below

    lookup_exprs: dict = {}
    for lu in ALL_LOOKUPS:
        exprs = [
            _subst(e, val)
            for e in (lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs)
        ]
        lookup_exprs[lu] = exprs

    # ── Backward liveness ───────────────────────────────────────────
    live: set = set()
    stack = [r for r in ROOT_DIMS if r not in val]
    const_roots = [r.name for r in ROOT_DIMS if r in val and hasattr(r, "name")]
    while stack:
        d = stack.pop()
        if d in live or d in val or isinstance(d, InputDimension):
            continue
        live.add(d)
        deps: set = set()
        if isinstance(d, ReGLUDimension):
            a, b = folded[d]
            deps |= set(a.terms) | set(b.terms)
        elif isinstance(d, PersistDimension):
            (e,) = folded[d]
            deps |= set(e.terms)
        elif isinstance(d, LookUpDimension):
            for e in lookup_exprs[d.lookup]:
                deps |= set(e.terms)
            # every dim of a live lookup rides along as produced
            for sib in d.lookup.dims:
                if sib not in live and sib not in val:
                    live.add(sib)
        for dep in deps:
            if dep not in val and not isinstance(dep, InputDimension):
                stack.append(dep)

    live_reglus = [d for d in live if isinstance(d, ReGLUDimension)]
    live_persists = [d for d in live if isinstance(d, PersistDimension)]
    live_lookups = {d.lookup for d in live if isinstance(d, LookUpDimension)}

    # ── Folded term count (nnz proxy) ───────────────────────────────
    terms = 0
    for d in live_reglus:
        a, b = folded[d]
        terms += len(a.terms) + len(b.terms)
    for d in live_persists:
        (e,) = folded[d]
        terms += len(e.terms)
    for lu in live_lookups:
        for e in lookup_exprs[lu]:
            terms += len(e.terms)

    # ── Depth on the pruned graph ───────────────────────────────────
    pruned_dims = [
        d for d in ALL_DIMS if isinstance(d, InputDimension) or d in live
    ]
    pruned_lus = [lu for lu in ALL_LOOKUPS if lu in live_lookups]
    pg = _build_graph(pruned_dims, pruned_lus, g.inv_log_pos)
    depth = _min_layers(pg["ops"], pg["op_deps"]) if pg["ops"] else 0

    return dict(
        label=label,
        reglus=len(live_reglus),
        persists=len(live_persists),
        lookups=len(live_lookups),
        terms=terms,
        depth=depth,
        const_roots=const_roots,
    )


def opcode_pins(selected: set[int]) -> dict[str, float]:
    """Pin every decode gate: 1 for a single selected opcode, 0 for the rest.
    Opcodes in `selected` with len>1 are left unpinned (family expert:
    the expert does its own decode among its members)."""
    pins: dict[str, float] = {"is_nmi": 0.0}
    for op in OPCODES.values():
        if op.byte in selected:
            if len(selected) == 1:
                pins[f"is_op_{op.byte:02X}"] = 1.0
            # else: leave unpinned -- expert decodes among its members
        else:
            pins[f"is_op_{op.byte:02X}"] = 0.0
    return pins


def main() -> None:
    results = {}

    t0 = time.time()
    base = analyze({}, "union")
    print(
        f"baseline (union graph): depth={base['depth']} layers, "
        f"terms={base['terms']}, reglus={base['reglus']}, "
        f"persists={base['persists']}, lookups={base['lookups']} "
        f"[{time.time() - t0:.1f}s]",
        file=sys.stderr,
    )
    results["union"] = base

    # NMI expert
    pins = {f"is_op_{op.byte:02X}": 0.0 for op in OPCODES.values()}
    pins["is_nmi"] = 1.0
    results["NMI"] = analyze(pins, "NMI")

    # Per-opcode experts
    per_opcode = {}
    t0 = time.time()
    for i, op in enumerate(OPCODES.values()):
        r = analyze(opcode_pins({op.byte}), f"{op.mnemonic} {op.mode} {op.byte:02X}")
        r["mnemonic"] = op.mnemonic
        r["operation"] = op.operation
        r["mode"] = op.mode
        r["byte"] = op.byte
        per_opcode[f"{op.byte:02X}"] = r
        if (i + 1) % 32 == 0:
            print(f"  opcode {i + 1}/{len(OPCODES)} [{time.time() - t0:.0f}s]", file=sys.stderr)
    results["per_opcode"] = per_opcode

    # Per-family experts (all modes of one operation share an expert)
    fam_members = defaultdict(set)
    for op in OPCODES.values():
        fam_members[op.operation].add(op.byte)
    per_family = {}
    t0 = time.time()
    for fam, members in fam_members.items():
        r = analyze(opcode_pins(members), f"family {fam}")
        r["n_opcodes"] = len(members)
        per_family[fam] = r
    print(f"families done [{time.time() - t0:.0f}s]", file=sys.stderr)
    results["per_family"] = per_family

    out = OUT_DIR / "moe_probe_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"wrote {out}", file=sys.stderr)

    # ── Summary ────────────────────────────────────────────────────
    def row(r, base):
        return (
            f"{r['label']:>28}  depth {r['depth']:>3} ({r['depth'] / base['depth']:.2f}x)  "
            f"terms {r['terms']:>6} ({r['terms'] / base['terms']:.3f}x)  "
            f"ops {r['reglus'] + r['persists'] + r['lookups']:>5}"
        )

    print("\n=== per-opcode experts (sorted by terms) ===")
    ops_sorted = sorted(per_opcode.values(), key=lambda r: r["terms"])
    for r in ops_sorted[:8]:
        print(row(r, base))
    print("  ...")
    for r in ops_sorted[-8:]:
        print(row(r, base))

    import statistics

    depths = [r["depth"] for r in per_opcode.values()]
    termss = [r["terms"] for r in per_opcode.values()]
    print(
        f"\nper-opcode: depth min/med/max = {min(depths)}/{int(statistics.median(depths))}/{max(depths)}"
        f"  terms min/med/max = {min(termss)}/{int(statistics.median(termss))}/{max(termss)}"
    )

    print("\n=== family experts (sorted by terms) ===")
    for r in sorted(per_family.values(), key=lambda r: r["terms"]):
        print(row(r, base))

    print("\n=== NMI expert ===")
    print(row(results["NMI"], base))
    print(f"\nbaseline: depth {base['depth']}, terms {base['terms']}")


if __name__ == "__main__":
    main()
