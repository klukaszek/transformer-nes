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
from transformer_nes.cpu6502.backends.evaluator import RUNTIME_READ_DIMS
from transformer_nes.cpu6502.backends.weights.windows import BRANCH_OPERATIONS
from transformer_nes.cpu6502.spec import OPCODES, READS_MEMORY
from transformer_nes.graph.fold import (
    SEQUENCE_FEEDBACK,
    GraphFolder,
    opcode_pins,
    sequence_pins,
    sequence_shape_pins,
)
from transformer_nes.model.weights import build_model
from transformer_nes.scheduler.milp import asap_schedule

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


def _eval_over_heads(fold_result, dim, length):
    """Evaluate a folded dim as a function of the symbolic head pc, for every
    head where the window fits without wrapping $FFFF.  The head enters the
    graph through the step-0 pc persists (state arrives via attention, so
    raw inputs never carry it); evaluation cuts there and raises if the
    value depends on anything else -- the straight-line license needs
    next_pc to be a pure function of pc."""
    from transformer_nes.graph.core import (
        InputDimension,
        LookUpDimension,
        PersistDimension,
        ReGLUDimension,
    )

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


def _decode_ops(code: bytes) -> list:
    ops, off = [], 0
    while off < len(code):
        op = OPCODES[code[off]]
        ops.append(op)
        off += op.length
    return ops


def _check_first_read_only(ops, label: str) -> None:
    """Only the first instruction of a window may read memory: the run
    loop's shadow-read watch check covers exactly one instruction."""
    for op in ops[1:]:
        if op.operation in READS_MEMORY and op.mode not in ("imp", "acc", "imm"):
            raise RuntimeError(
                f"{label}: {op.mnemonic} reads memory past the shadow-watch "
                "check (only the first instruction may)"
            )


def _const_root(fold_result, root: str, k: int, label: str) -> int:
    one = fold_result.program_graph.one
    d = fold_result.dim_of_name[f"{root}__s{k}"]
    expr = getattr(d, "expr", None)
    if expr is None or any(t is not one for t in expr.terms):
        raise RuntimeError(
            f"{label} step {k}: {root} did not fold to a constant"
        )
    return int(round(expr.terms.get(one, 0.0)))


def _check_write_free(fold_result, n_steps: int, label: str) -> None:
    """Every step of a routed window must be write-free and non-halting
    (verified on the folded constants, not assumed from the opcode list)."""
    for k in range(n_steps):
        for store in ("is_store", "is_store2", "is_store3"):
            if _const_root(fold_result, store, k, label) != 0:
                raise RuntimeError(f"{label} step {k}: not write-free")
        if _const_root(fold_result, "new_halted", k, label) != 0:
            raise RuntimeError(f"{label} step {k}: may halt")


def _schedule_and_build(fold_result):
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
            from transformer_nes.scheduler.milp import milp_schedule

            t0 = time.perf_counter()
            asap = asap_schedule(fp.input_tokens, fp.output_tokens,
                                 program_graph=fp, log=logger.debug)
            _m, _t, _ti, ainfo = build_model(program_graph=fp, schedule=asap)
            sched = milp_schedule(fp.input_tokens, fp.output_tokens,
                                  program_graph=fp, log=logger.debug,
                                  time_limit=milp_seconds,
                                  objective="hygiene",
                                  width_bound=ainfo.d_model)
            logger.info("expert %02X: hygiene MILP in %.0fs",
                        byte, time.perf_counter() - t0)
            model, _tokens, tok_to_idx, info = build_model(
                program_graph=fp, schedule=sched,
            )
        else:
            model, _tokens, tok_to_idx, info = _schedule_and_build(r)
        experts.append({
            "name": f"{byte:02X}", "byte": byte, "model": model,
            "step_spec": fold_step_spec(r, model, tok_to_idx, info),
            "union_lookup_of": r.union_lookup_of,
            "_layers": info.n_layers, "_d": info.d_model,
        })
    logger.info(
        "routed experts: %d rebuilt models, %d-%d layers, d_model %d-%d",
        len(experts),
        min(e["_layers"] for e in experts), max(e["_layers"] for e in experts),
        min(e["_d"] for e in experts), max(e["_d"] for e in experts),
    )
    return experts


def build_sequence_expert(pg, head: int, code: bytes, repeat: int) -> dict:
    """One sequence expert: a K-instruction window folded as ONE forward.

    The router guarantees pc == head, the window's bytes, no pending NMI,
    and a remaining step/cycle budget covering the whole window, so the
    fold pins all of it.  The license: every step write-free and
    non-halting, control flow straight-line inside the window (folded
    next_pc == the next step's pinned pc), and only the first instruction
    may read memory.
    """
    label = f"sequence @{head:04X}"
    pin_list = sequence_pins(head, code, OPCODES) * repeat
    n_steps = len(pin_list)
    ops = _decode_ops(code) * repeat
    _check_first_read_only(ops, label)

    folder = GraphFolder(pg, root_names=RUNTIME_READ_DIMS)
    r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK,
                             label=f"seq_{head:04X}")
    if r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions "
            f"of {r.insert_side_hazards}"
        )
    _check_write_free(r, n_steps, label)
    for k in range(n_steps - 1):
        nxt = int(pin_list[k + 1]["current_pc_lo"]) | (
            int(pin_list[k + 1]["current_pc_hi"]) << 8
        )
        if _const_root(r, "next_pc", k, label) != nxt:
            raise RuntimeError(
                f"{label} step {k}: control flow leaves the window "
                "(next_pc != the next step's pc)"
            )

    model, _tokens, tok_to_idx, info = _schedule_and_build(r)
    cycles = sum(op.cycles for op in ops)
    return {
        "kind": "seq", "name": f"seq_{head:04X}", "head": head,
        "window": bytes(code), "n_steps": n_steps, "cycles": cycles,
        "cycles_prefix": cycles - ops[-1].cycles, "model": model,
        "step_spec": fold_step_spec(r, model, tok_to_idx, info,
                                    n_steps=n_steps),
        "union_lookup_of": r.union_lookup_of,
        "_layers": info.n_layers, "_d": info.d_model,
    }


def build_shape_experts(pg, shapes) -> tuple[list[dict], list[dict]]:
    """Canonical shape experts for the runtime window table.

    Each shape folds ONCE with pc and every operand byte symbolic
    (fold_sequence pc_offsets); the license: only the first instruction may
    read memory, every step write-free and non-halting, and each interior
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
        ops = _decode_ops(template)
        offs = [0]
        for op in ops[:-1]:
            offs.append(offs[-1] + op.length)
        checks = list(zip(offs, (op.byte for op in ops), strict=True))
        _check_first_read_only(ops, label)
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
        _check_write_free(r, n_steps, label)
        if not self_target:
            for k in range(n_steps - 1):
                heads, got = _eval_over_heads(
                    r, r.dim_of_name[f"next_pc__s{k}"], len(template)
                )
                if not np.array_equal(got, heads + pc_offsets[k + 1]):
                    raise RuntimeError(
                        f"{label} step {k}: interior next_pc is not "
                        "head + offset for every head"
                    )

        model, _tokens, tok_to_idx, info = _schedule_and_build(r)
        ops_full = ops * repeat
        cycles = sum(op.cycles for op in ops_full)
        experts.append({
            "kind": "shape", "name": f"shape_{name}", "shape_id": sid,
            "opcode_checks": checks, "length": len(template),
            "self_target": self_target, "n_steps": n_steps,
            "cycles_prefix": cycles - ops_full[-1].cycles, "model": model,
            "step_spec": fold_step_spec(r, model, tok_to_idx, info,
                                        n_steps=n_steps),
            "union_lookup_of": r.union_lookup_of,
            "_layers": info.n_layers, "_d": info.d_model,
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
            min(e["_layers"] for e in experts), max(e["_layers"] for e in experts),
            min(e["_d"] for e in experts), max(e["_d"] for e in experts),
        )
    return experts, meta
