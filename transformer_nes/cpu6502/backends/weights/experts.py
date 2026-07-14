"""Fold-and-build expert models for the generated weights engine.

Each builder partially evaluates the union graph (graph/fold.py), checks its
dispatch license (checked, never assumed), and compiles the folded graph into
its own transformer.  The machine (weights_runner) owns orchestration and the
engine cache; this module owns folds, licenses, and model construction.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS
from transformer_nes.cpu6502.backends.weights.windows import (
    BRANCH_OPERATIONS,
    MEMORY_WRITE_OPERATIONS,
)
from transformer_nes.cpu6502.spec import OPCODES, READS_MEMORY
from transformer_nes.graph.core import (
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
)
from transformer_nes.graph.fold import (
    SEQUENCE_FEEDBACK,
    GraphFolder,
    opcode_pins,
    sequence_pins,
    sequence_shape_pins,
)
from transformer_nes.model.weights import build_model
from transformer_nes.scheduler.milp import asap_schedule, milp_schedule

logger = logging.getLogger(__name__)

# Residual input slots shared by every generated step spec: spec key -> the
# graph dimension name it resolves through (union slot map or fold name map).
INPUT_SLOT_NAMES = {
    "is_state": "is_state",
    "pc_lo": "in_pc_lo", "pc_hi": "in_pc_hi",
    "a": "in_a", "x": "in_x", "y": "in_y", "s": "in_s",
    "ir": "in_ir", "oplo": "in_oplo", "ophi": "in_ophi",
    "ea_lo": "in_ea_lo", "ea_hi": "in_ea_hi",
    "data": "in_data", "halted": "in_halted", "nmi": "in_nmi",
    "irq": "in_irq",
}
P_FLAG_NAMES = (
    ("in_p_c", reference.FLAG_C), ("in_p_z", reference.FLAG_Z),
    ("in_p_i", reference.FLAG_I), ("in_p_d", reference.FLAG_D),
    ("in_p_b", reference.FLAG_B), ("in_p_u", reference.FLAG_U),
    ("in_p_v", reference.FLAG_V), ("in_p_n", reference.FLAG_N),
)


def head_row(model, row: int) -> tuple[list[int], list[float]]:
    """One output-head row as sparse (columns, values) lists."""
    w = model.head.weight[row]
    cols = np.nonzero(w)[0]
    return ([int(c) for c in cols], [float(w[c]) for c in cols])


def step_spec_base(slot) -> dict:
    """The input/p_flag half of a generated step spec; ``slot`` maps a graph
    dimension name to its residual slot."""
    return {
        "inputs": {key: slot(name) for key, name in INPUT_SLOT_NAMES.items()},
        "p_flags": [(slot(name), mask) for name, mask in P_FLAG_NAMES],
    }


def fold_step_spec(fold_result, model, tok_to_idx, info, *,
                   n_steps: int | None = None) -> dict:
    """Slot map for one folded model's generated step, resolved through the
    fold's name map and the model's own schedule/slot assignment.  ``n_steps``
    switches to per-step output maps (sequence/shape folds)."""

    def slot(name):
        return info.slot_of[fold_result.dim_of_name[name]]

    spec = step_spec_base(slot)
    if n_steps is None:
        spec["outputs"] = {name: slot(name) for name in RUNTIME_READ_DIMS}
    else:
        spec["step_outputs"] = [
            {name: slot(f"{name}__s{k}") for name in RUNTIME_READ_DIMS}
            for k in range(n_steps)
        ]
    spec["halt_head"] = head_row(model, tok_to_idx["HALT"])
    spec["state_head"] = head_row(model, tok_to_idx["STATE"])
    return spec


def eval_over_heads(fold_result, dim, length):
    """Evaluate a folded dim as a function of the symbolic head pc, for every
    head where the window fits without wrapping $FFFF.  The head enters the
    graph through the step-0 pc persists (state arrives via attention, so
    raw inputs never carry it); evaluation cuts there and raises if the
    value depends on anything else -- the straight-line license needs
    next_pc to be a pure function of pc."""
    heads = np.arange(0x10000 - length + 1, dtype=np.float64)
    memo: dict = {
        fold_result.dim_of_name["current_pc_lo__s0"]: heads % 256.0,
        fold_result.dim_of_name["current_pc_hi__s0"]: heads // 256.0,
    }

    def ev(d):
        r = memo.get(d)
        if r is not None:
            return r
        if isinstance(d, InputDimension):
            if d.name != "one":
                raise RuntimeError(f"depends on input {d.name!r}")
            r = np.float64(1.0)
        elif isinstance(d, ReGLUDimension):
            a, b = np.asarray(evx(d.a_expr)), np.asarray(evx(d.b_expr))
            r = np.where(b > 0.0, a * b, 0.0)
        elif isinstance(d, PersistDimension):
            r = evx(d.expr)
        elif isinstance(d, LookUpDimension):
            raise RuntimeError("depends on a memory lookup")
        else:
            raise RuntimeError(f"unsupported dim {d.name!r}")
        memo[d] = r
        return r

    def evx(e):
        acc = np.float64(0.0)
        for d, c in e.terms.items():
            acc = acc + ev(d) * c
        return acc

    return heads, np.asarray(ev(dim))


def decode_ops(code: bytes) -> list:
    ops, off = [], 0
    while off < len(code):
        op = OPCODES[code[off]]
        ops.append(op)
        off += op.length
    return ops


def check_reads_licensed(fold_result, ops, label: str) -> None:
    """The run loop shadow-checks only the first instruction.  A later
    memory read is nevertheless safe when folding proves its effective
    address stays in native RAM (< $2000) or cartridge ROM (>= $8000),
    outside every NES device/read-watch range."""
    wrote_memory = ops[0].operation in MEMORY_WRITE_OPERATIONS
    for k, op in enumerate(ops[1:], 1):
        if op.operation in READS_MEMORY and op.mode not in ("imp", "acc", "imm"):
            if wrote_memory:
                raise RuntimeError(
                    f"{label}: {op.mnemonic} step {k} reads after an earlier "
                    "write that is not fed back inside one forward"
                )
            dim = fold_result.dim_of_name[f"ea__s{k}"]
            lo, hi = interval_of(fold_result, dim)
            if hi < 0x2000 or lo >= 0x8000:
                continue
            raise RuntimeError(
                f"{label}: {op.mnemonic} step {k} read target bound "
                f"[{lo:.0f}, {hi:.0f}] may reach a watched device"
            )
        wrote_memory |= op.operation in MEMORY_WRITE_OPERATIONS


def const_root(fold_result, root: str, k: int, label: str) -> int:
    one = fold_result.program_graph.one
    d = fold_result.dim_of_name[f"{root}__s{k}"]
    expr = getattr(d, "expr", None)
    if expr is None or any(t is not one for t in expr.terms):
        raise RuntimeError(
            f"{label} step {k}: {root} did not fold to a constant"
        )
    return int(round(expr.terms.get(one, 0.0)))


def interval_of(fold_result, dim) -> tuple[float, float]:
    """Conservative [lo, hi] bound of a folded dim over byte-valued inputs
    (bits give [0, 1], 'one' gives 1); raises on memory lookups, whose
    values cannot be bounded ahead of dispatch."""
    BIT_INPUTS = frozenset({
        "is_state", "in_nmi", "in_irq", "in_halted",
        "in_p_c", "in_p_z", "in_p_i", "in_p_d",
        "in_p_b", "in_p_u", "in_p_v", "in_p_n",
    })
    memo: dict = {}

    def ev(d):
        r = memo.get(d)
        if r is not None:
            return r
        if isinstance(d, InputDimension):
            if d.name == "one":
                r = (1.0, 1.0)
            elif d.name in BIT_INPUTS:
                r = (0.0, 1.0)
            else:
                r = (0.0, 255.0)
        elif isinstance(d, ReGLUDimension):
            (alo, ahi), bhi = evx(d.a_expr), evx(d.b_expr)[1]
            if bhi <= 0.0:
                r = (0.0, 0.0)
            else:
                # b in (0, bhi]: extremes at b -> 0 and b == bhi.
                r = (min(0.0, alo * bhi, ahi * bhi),
                     max(0.0, alo * bhi, ahi * bhi))
        elif isinstance(d, PersistDimension):
            r = evx(d.expr)
        elif isinstance(d, LookUpDimension):
            # A lookup returns one of its insert-side values (registers
            # come in via state attention, so this is the live data
            # path), or 0 when nothing matched.
            lo, hi = evx(d.lookup.value_exprs[d.value_index])
            r = (min(lo, 0.0), max(hi, 0.0))
        else:
            raise RuntimeError(f"unbounded dim {d.name!r}")
        memo[d] = r
        return r

    def evx(e):
        lo = hi = 0.0
        for d, c in e.terms.items():
            dlo, dhi = ev(d)
            lo += min(c * dlo, c * dhi)
            hi += max(c * dlo, c * dhi)
        return (lo, hi)

    return ev(dim)


STORE_SLOTS = (
    ("is_store", "write_addr_out"),
    ("is_store2", "write_addr2_out"),
    ("is_store3", "write_addr3_out"),
)


def check_writes_licensed(fold_result, n_steps: int, label: str, *,
                          window_range: tuple[int, int] | None = None) -> bool:
    """Stores are licensed when the folded write address provably stays
    below $2000 -- the console only ever watches registers at or above it,
    and dispatch additionally requires no watch range below $2000, so a
    licensed store can never swallow a watch_write pause.  window_range
    (pinned windows) additionally forbids overlap with the window's own
    code bytes: the fold pinned them, so a self-modifying store would
    execute stale semantics.  Every step must still be non-halting.
    Returns whether any step stores."""
    has_stores = False
    for k in range(n_steps):
        for store, addr in STORE_SLOTS:
            if const_root(fold_result, store, k, label) == 0:
                continue
            has_stores = True
            dim = fold_result.dim_of_name[f"{addr}__s{k}"]
            try:
                lo, hi = interval_of(fold_result, dim)
            except RuntimeError as e:
                raise RuntimeError(f"{label} step {k}: store target {e}") from e
            if lo < 0.0 or hi >= 0x2000:
                raise RuntimeError(
                    f"{label} step {k}: store target bound [{lo:.0f}, {hi:.0f}]"
                    " is not provably below $2000"
                )
            if window_range is not None:
                w_lo, w_hi = window_range
                if hi >= w_lo and lo < w_hi:
                    raise RuntimeError(
                        f"{label} step {k}: store bound [{lo:.0f}, {hi:.0f}] "
                        "may modify the window's own pinned bytes"
                    )
        if const_root(fold_result, "new_halted", k, label) != 0:
            raise RuntimeError(f"{label} step {k}: may halt")
    return has_stores


def schedule_and_build(fold_result):
    fp = fold_result.program_graph
    sched = asap_schedule(fp.input_tokens, fp.output_tokens,
                          program_graph=fp, log=logger.debug)
    return build_model(program_graph=fp, schedule=sched)


def build_opcode_experts(pg, opcodes, *, schedule: str = "asap",
                         milp_seconds: int = 900) -> list[dict]:
    """Rebuilt per-opcode expert models for the opcode-routed engine.

    For each expert opcode, the union graph is partially evaluated with the
    decode one-hot gates pinned (is_op_<byte>=1, every other is_op_*=0,
    is_nmi=0 -- exactly what the hard router supplies), and the *folded*
    graph is scheduled and weight-compiled into its own transformer.

    Safety is checked, not assumed: the fold refuses lookups whose
    insert-side expressions can reach a pin (shared attention state must be
    pin-independent), and engine_gen refuses any expert head whose
    classification disagrees with the union's.  Correctness rests on
    ram[pc] == the graph's fetched opcode; the lockstep chain (every opcode
    swept through its expert by nestest and Klaus) is the receipt.
    """
    opcodes = [int(b) for b in (opcodes or []) if int(b) in OPCODES]
    if not opcodes:
        return []

    folder = GraphFolder(pg, root_names=RUNTIME_READ_DIMS)
    experts = []
    for byte in opcodes:
        r = folder.fold(opcode_pins(byte, OPCODES), label=f"{byte:02X}")
        if r.insert_side_hazards:
            raise RuntimeError(
                f"expert {byte:02X}: decode pins reach insert-side "
                f"expressions of {r.insert_side_hazards}; shared attention "
                "state would diverge"
            )
        fp = r.program_graph
        if schedule == "milp":
            # Exact hygiene-objective schedule: tractable at expert scale; at
            # the time limit the solver's best incumbent is used -- feasible,
            # just unproven.  Width is capped at the balanced-ASAP width.
            t0 = time.perf_counter()
            asap = asap_schedule(fp.input_tokens, fp.output_tokens,
                                 program_graph=fp, log=logger.debug)
            _m, _t, _ti, ainfo = build_model(program_graph=fp, schedule=asap)
            sched = milp_schedule(fp.input_tokens, fp.output_tokens,
                                  program_graph=fp, log=logger.debug,
                                  time_limit=milp_seconds,
                                  width_bound=ainfo.d_model)
            logger.info("expert %02X: hygiene MILP in %.0fs",
                        byte, time.perf_counter() - t0)
            model, _tokens, tok_to_idx, info = build_model(
                program_graph=fp, schedule=sched,
            )
        else:
            model, _tokens, tok_to_idx, info = schedule_and_build(r)
        experts.append({
            "name": f"{byte:02X}", "byte": byte, "model": model,
            "step_spec": fold_step_spec(r, model, tok_to_idx, info),
            "union_lookup_of": r.union_lookup_of,
            "layers": info.n_layers, "_d": info.d_model,
        })
    logger.info(
        "routed experts: %d rebuilt models, %d-%d layers, d_model %d-%d",
        len(experts),
        min(e["layers"] for e in experts), max(e["layers"] for e in experts),
        min(e["_d"] for e in experts), max(e["_d"] for e in experts),
    )
    return experts


def build_sequence_expert(pg, head: int, code: bytes, repeat: int) -> dict:
    """One sequence expert: a K-instruction window folded as ONE forward.

    The router guarantees pc == head, the window's bytes, no pending NMI,
    and a remaining step/cycle budget covering the whole window, so the
    fold pins all of it.  The license: every step non-halting with stores
    bounded below $2000 (check_writes_licensed), control flow
    straight-line inside the window (folded next_pc == the next step's
    pinned pc), and later reads proven outside watched device ranges.
    """
    label = f"sequence @{head:04X}"
    pin_list = sequence_pins(head, code, OPCODES) * repeat
    n_steps = len(pin_list)
    ops = decode_ops(code) * repeat
    folder = GraphFolder(pg, root_names=RUNTIME_READ_DIMS)
    r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK,
                             label=f"seq_{head:04X}")
    if r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions "
            f"of {r.insert_side_hazards}"
        )
    check_reads_licensed(r, ops, label)
    has_stores = check_writes_licensed(
        r, n_steps, label, window_range=(head, head + len(code)),
    )
    for k in range(n_steps - 1):
        nxt = int(pin_list[k + 1]["current_pc_lo"]) | (
            int(pin_list[k + 1]["current_pc_hi"]) << 8
        )
        if const_root(r, "next_pc", k, label) != nxt:
            raise RuntimeError(
                f"{label} step {k}: control flow leaves the window "
                "(next_pc != the next step's pc)"
            )

    model, _tokens, tok_to_idx, info = schedule_and_build(r)
    cycles = sum(op.cycles for op in ops)
    # Window bytes join the name: different banks may carry different
    # windows at one head, and each needs its own step_core symbol.
    return {
        "kind": "seq", "name": f"seq_{head:04X}_{bytes(code).hex()}",
        "head": head, "has_stores": has_stores,
        "window": bytes(code), "n_steps": n_steps, "cycles": cycles,
        "cycles_prefix": cycles - ops[-1].cycles, "model": model,
        "step_spec": fold_step_spec(r, model, tok_to_idx, info,
                                    n_steps=n_steps),
        "union_lookup_of": r.union_lookup_of,
        "layers": info.n_layers, "_d": info.d_model,
    }


def build_shape_experts(pg, shapes) -> tuple[list[dict], list[dict]]:
    """Canonical shape experts for the runtime window table.

    Each shape folds ONCE with pc and every operand byte symbolic
    (fold_sequence pc_offsets); the license: only the first instruction may
    read memory, every step non-halting with stores bounded below $2000
    (a symbolic zero-page operand bounds itself), and each interior
    step's next_pc equals head + offset -- verified EXHAUSTIVELY over every
    head the window fits at, which is possible exactly because the folded
    expression may depend on nothing but the pc bytes.  The JMP self-spin's
    closure is a dispatch property instead (the router compares the operand
    bytes to the head).  A library shape that fails its license is a hard
    error: the library is code.
    """
    folder = GraphFolder(pg, root_names=RUNTIME_READ_DIMS)
    experts: list[dict] = []
    meta: list[dict] = []
    for sid, (name, template, repeat) in enumerate(shapes):
        label = f"shape {name}"
        ops = decode_ops(template)
        offs = [0]
        for op in ops[:-1]:
            offs.append(offs[-1] + op.length)
        checks = list(zip(offs, (op.byte for op in ops), strict=True))
        self_target = (
            len(ops) == 1 and ops[0].operation == "jmp" and ops[0].mode == "abs"
        )
        if repeat > 1 and not self_target:
            raise RuntimeError(f"{label}: only a JMP self-spin may repeat")
        if not self_target and ops[-1].operation not in BRANCH_OPERATIONS:
            raise RuntimeError(f"{label}: window must close via its last op")

        pins, offsets = sequence_shape_pins(template, OPCODES)
        pin_list = pins * repeat
        pc_offsets = offsets * repeat
        n_steps = len(pin_list)
        r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK,
                                 label=f"shape_{name}", pc_offsets=pc_offsets)
        if r.insert_side_hazards:
            raise RuntimeError(
                f"{label}: pins reach insert-side expressions "
                f"of {r.insert_side_hazards}"
            )
        check_reads_licensed(r, ops * repeat, label)
        has_stores = check_writes_licensed(r, n_steps, label)
        if has_stores:
            # A shape's head is symbolic, so store disjointness from the
            # window's own bytes cannot be proven; store idioms stay
            # pinned-window-only until a head-range dispatch guard exists.
            raise RuntimeError(f"{label}: store shapes are not licensed")
        if not self_target:
            for k in range(n_steps - 1):
                heads, got = eval_over_heads(
                    r, r.dim_of_name[f"next_pc__s{k}"], len(template)
                )
                if not np.array_equal(got, heads + pc_offsets[k + 1]):
                    raise RuntimeError(
                        f"{label} step {k}: interior next_pc is not "
                        "head + offset for every head"
                    )

        model, _tokens, tok_to_idx, info = schedule_and_build(r)
        ops_full = ops * repeat
        cycles = sum(op.cycles for op in ops_full)
        experts.append({
            "kind": "shape", "name": f"shape_{name}", "shape_id": sid,
            "opcode_checks": checks, "length": len(template),
            "self_target": self_target, "n_steps": n_steps,
            "has_stores": has_stores,
            "cycles_prefix": cycles - ops_full[-1].cycles, "model": model,
            "step_spec": fold_step_spec(r, model, tok_to_idx, info,
                                        n_steps=n_steps),
            "union_lookup_of": r.union_lookup_of,
            "layers": info.n_layers, "_d": info.d_model,
        })
        meta.append({
            "id": sid, "name": name, "sig": [[o, b] for o, b in checks],
            "length": len(template), "repeat": repeat,
            "close": "jmp" if self_target else "branch",
        })
    if experts:
        logger.info(
            "shape experts: %d folded, %d-%d layers, d_model %d-%d",
            len(experts),
            min(e["layers"] for e in experts), max(e["layers"] for e in experts),
            min(e["_d"] for e in experts), max(e["_d"] for e in experts),
        )
    return experts, meta
