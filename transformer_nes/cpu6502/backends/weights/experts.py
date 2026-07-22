"""Fold-and-build expert models for the generated weights engine.

Each builder partially evaluates the union graph (graph/fold.py), checks its
dispatch license (checked, never assumed), and compiles the folded graph into
its own transformer.  The machine (weights_runner) owns orchestration and the
engine cache; this module owns folds, licenses, and model construction.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import time

import numpy as np

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS
from transformer_nes.cpu6502.backends.weights.windows import (
    BRANCH_OPERATIONS,
    MEMORY_WRITE_OPERATIONS,
    SEQUENCE_WRITERS,
)
from transformer_nes.cpu6502.spec import OPCODES, READS_MEMORY
from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    project_program_graph,
)
from transformer_nes.graph.execution import (
    ArchitecturalOutput,
    ControlFlow,
    ExecutionGraph,
    MemoryWrite,
    StateField,
)
from transformer_nes.graph.fold import (
    SEQUENCE_FEEDBACK,
    GraphFolder,
    SequenceRejection,
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
# wide_bits builds only (absent from other graphs, so added when present):
# served A-bit state inputs, MSB first.
OPTIONAL_INPUT_SLOT_NAMES = {
    f"a_bit{b}": f"in_a_bit{b}" for b in range(7, -1, -1)
}


def head_row(model, row: int) -> tuple[list[int], list[float]]:
    """One output-head row as sparse (columns, values) lists."""
    w = model.head.weight[row]
    cols = np.nonzero(w)[0]
    return ([int(c) for c in cols], [float(w[c]) for c in cols])


def step_spec_base(slot) -> dict:
    """The input/p_flag half of a generated step spec; ``slot`` maps a graph
    dimension name to its residual slot."""
    spec = {"inputs": {}, "p_flags": []}
    for key, name in INPUT_SLOT_NAMES.items():
        with contextlib.suppress(KeyError):
            spec["inputs"][key] = slot(name)
    for name, mask in P_FLAG_NAMES:
        with contextlib.suppress(KeyError):
            spec["p_flags"].append((slot(name), mask))
    for key, name in OPTIONAL_INPUT_SLOT_NAMES.items():
        with contextlib.suppress(KeyError):
            spec["inputs"][key] = slot(name)
    return spec


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


SEQUENCE_STATE_DIMS = (
    "new_a", "new_x", "new_y", "new_s", "new_p", "new_halted",
    "fetched_opcode", "fetched_operand_lo", "fetched_operand_hi",
    "ea", "data_out",
)

SEQUENCE_PRESERVE_INPUTS = {
    "new_a": (("in_a", 1),),
    "new_x": (("in_x", 1),),
    "new_y": (("in_y", 1),),
    "new_s": (("in_s", 1),),
    "new_p": tuple(
        ("one" if mask == reference.FLAG_U else name, mask)
        for name, mask in P_FLAG_NAMES
    ),
    "new_halted": (("in_halted", 1),),
    "fetched_opcode": (("in_ir", 1),),
    "fetched_operand_lo": (("in_oplo", 1),),
    "fetched_operand_hi": (("in_ophi", 1),),
    "ea": (("in_ea_lo", 1), ("in_ea_hi", 256)),
    "data_out": (("in_data", 1),),
}

WRITE_RECEIPTS = (
    ("is_store", "write_addr_out", "write_value_out"),
    ("is_store2", "write_addr2_out", "write_value2_out"),
    ("is_store3", "write_addr3_out", "write_value3_out"),
)
SEQUENCE_ROOT_DIMS = (*RUNTIME_READ_DIMS, "branch_taken")


def linear_input_form(expr: Expression) -> dict:
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


def classify_sequence_state(fold_result, final_names) -> dict:
    by_name = fold_result.dim_of_name
    one = fold_result.program_graph.one
    result = {"model": {}, "preserve": [], "constant": {}}
    for name, dim_name in final_names.items():
        dim = by_name[dim_name]
        form = linear_input_form(Expression({dim: 1.0}))
        expected = {
            by_name[input_name]: float(coeff)
            for input_name, coeff in SEQUENCE_PRESERVE_INPUTS[name]
        }
        if form == expected:
            result["preserve"].append(name)
        elif all(source is one for source in form):
            result["constant"][name] = int(round(form.get(one, 0.0)))
        else:
            result["model"][name] = dim_name
    return result


def project_sequence_execution(
    fold_result, n_steps: int, control: ControlFlow,
) -> ExecutionGraph:
    """Replace the proof graph's per-step trace ABI with its execution ABI.

    The complete fold is retained until all control, read, write and halt
    licenses have passed.  Only then do we keep the final architectural
    state and the address/value of each statically-live ordered write.  The
    projected graph is what gets scheduled and converted to weights.
    """
    final_names = {
        name: f"{name}__s{n_steps - 1}" for name in SEQUENCE_STATE_DIMS
    }
    final = classify_sequence_state(fold_result, final_names)
    outputs = []
    output_tokens = {}
    for name in SEQUENCE_STATE_DIMS:
        if name in final["model"]:
            dim = fold_result.dim_of_name[final["model"][name]]
            outputs.append(ArchitecturalOutput(name, "model", dimension=dim))
            output_tokens[f"state:{name}"] = Expression({dim: 1})
        elif name in final["constant"]:
            outputs.append(ArchitecturalOutput(
                name, "constant", constant=final["constant"][name],
            ))
        else:
            outputs.append(ArchitecturalOutput(name, "preserve"))

    writes = []
    for k in range(n_steps):
        for order, (flag, addr, value) in enumerate(WRITE_RECEIPTS):
            active = const_root(fold_result, flag, k, fold_result.label)
            if active not in (0, 1):
                raise RuntimeError(
                    f"{fold_result.label} s{k}: {flag} is not boolean"
                )
            if not active:
                continue
            address = fold_result.dim_of_name[f"{addr}__s{k}"]
            write_value = fold_result.dim_of_name[f"{value}__s{k}"]
            writes.append(MemoryWrite(k, order, address, write_value))
            output_tokens[f"write:{k}:{order}:address"] = Expression({address: 1})
            output_tokens[f"write:{k}:{order}:value"] = Expression({write_value: 1})

    if control.predicate is not None:
        output_tokens["control:branch"] = Expression({control.predicate: 1})

    program_graph = project_program_graph(
        fold_result.program_graph, output_tokens, retain_input_tokens=False,
    )
    dim_of_name = {}
    for dim in program_graph.all_dims:
        dim_of_name.setdefault(dim.name, dim)
    state_fields = tuple(
        StateField(dim.name, dim)
        for dim in program_graph.all_dims
        if isinstance(dim, InputDimension) and dim is not program_graph.one
    )
    return ExecutionGraph(
        program_graph=program_graph,
        n_steps=n_steps,
        state_fields=state_fields,
        memory_reads=tuple(program_graph.all_memory_reads),
        memory_writes=tuple(writes),
        outputs=tuple(outputs),
        control=control,
        dim_of_name=dim_of_name,
    )


def sequence_step_spec(execution: ExecutionGraph, info) -> dict:
    def slot(name):
        return info.slot_of[execution.dim_of_name[name]]

    spec = step_spec_base(slot)
    spec["sequence_final"] = {
        "model": {
            output.name: info.slot_of[output.dimension]
            for output in execution.outputs if output.source == "model"
        },
        "preserve": [
            output.name for output in execution.outputs
            if output.source == "preserve"
        ],
        "constant": {
            output.name: output.constant for output in execution.outputs
            if output.source == "constant"
        },
    }
    writes = [[] for _ in range(execution.n_steps)]
    for write in execution.memory_writes:
        writes[write.step].append((
            info.slot_of[write.address], info.slot_of[write.value],
        ))
    spec["sequence_writes"] = writes
    control = execution.control
    spec["sequence_control"] = {
        "predicate": (
            None if control.predicate is None
            else info.slot_of[control.predicate]
        ),
        "fallthrough": control.fallthrough,
        "target": control.target,
        "relative_to_head": control.relative_to_head,
        "relative_operand_offset": control.relative_operand_offset,
    }
    return spec


def eval_over_heads(fold_result, dim, length):
    """Evaluate a folded dim as a function of the symbolic head pc, for every
    head where the window fits without wrapping $FFFF.  The head enters the
    graph through the step-0 pc persists (state arrives via attention, so
    raw inputs never carry it); evaluation cuts there and raises if the
    value depends on anything else -- the straight-line license needs
    next_pc to be a pure function of pc."""
    heads = np.arange(0x10000 - length + 1, dtype=np.float64)
    if "current_pc_lo__s0" in fold_result.dim_of_name:
        memo: dict = {
            fold_result.dim_of_name["current_pc_lo__s0"]: heads % 256.0,
            fold_result.dim_of_name["current_pc_hi__s0"]: heads // 256.0,
        }
    else:
        # Linear-inlined shape folds may substitute the byte persists into
        # the already-combined address while retaining the same symbolic
        # head license.
        memo = {fold_result.dim_of_name["current_pc__s0"]: heads}

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


# Operations whose folded circuits read A's bit state (wide_bits builds):
# logical/bitwise families, ADC/SBC (served-sign overflow + decimal
# nibbles), and any accumulator-mode shift (operand bits mux to A's).
A_BIT_OPERATIONS = frozenset({
    "and", "ora", "eor", "bit", "sax", "slo", "rla", "sre", "adc", "sbc",
})


def last_a_bit_step(ops) -> int:
    """Index of the last step consuming A's bit state; the fold seeds the
    bit feedback chain only strictly before it (wrong pruning fails
    loudly at build, not silently)."""
    last = -1
    for k, op in enumerate(ops):
        if op.operation in A_BIT_OPERATIONS or op.mode == "acc":
            last = k
    return last


def decode_ops(code: bytes) -> list:
    ops, off = [], 0
    while off < len(code):
        op = OPCODES[code[off]]
        ops.append(op)
        off += op.length
    return ops


def pinned_sequence_control(fold_result, pin_list, ops) -> ControlFlow:
    k = len(pin_list) - 1
    op = ops[k]
    pins = pin_list[k]
    pc = int(pins["current_pc_lo"]) | (int(pins["current_pc_hi"]) << 8)
    fallthrough = (pc + op.length) & 0xFFFF
    if op.operation in BRANCH_OPERATIONS:
        offset = int(pins["fetched_operand_lo_all"])
        signed = offset - 256 if offset & 0x80 else offset
        predicate = fold_result.dim_of_name[f"branch_taken__s{k}"]
        form = linear_input_form(Expression({predicate: 1}))
        one = fold_result.program_graph.one
        if all(dim is one for dim in form):
            successor = (
                (fallthrough + signed) & 0xFFFF
                if int(round(form.get(one, 0.0))) else fallthrough
            )
            return ControlFlow(
                predicate=None, fallthrough=successor, target=successor,
            )
        return ControlFlow(
            predicate=predicate,
            fallthrough=fallthrough,
            target=(fallthrough + signed) & 0xFFFF,
        )
    if op.operation == "jmp" and op.mode == "abs":
        target = int(pins["fetched_operand_lo_all"]) | (
            int(pins["fetched_operand_hi_all"]) << 8
        )
        return ControlFlow(predicate=None, fallthrough=target, target=target)
    if op.operation in SEQUENCE_WRITERS or op.operation == "jmp":
        raise RuntimeError(
            f"sequence final control {op.mnemonic} is not host-resolvable"
        )
    return ControlFlow(
        predicate=None, fallthrough=fallthrough, target=fallthrough,
    )


def shape_sequence_control(
    fold_result, n_steps: int, ops, offsets, length: int, self_target: bool,
) -> ControlFlow:
    if self_target:
        return ControlFlow(
            predicate=None, fallthrough=0, target=0, relative_to_head=True,
        )
    op = ops[-1]
    if op.operation in BRANCH_OPERATIONS:
        return ControlFlow(
            predicate=fold_result.dim_of_name[f"branch_taken__s{n_steps - 1}"],
            fallthrough=length,
            target=0,
            relative_to_head=True,
            relative_operand_offset=offsets[-1] + 1,
        )
    if op.operation in SEQUENCE_WRITERS or op.operation == "jmp":
        raise RuntimeError(
            f"shape final control {op.mnemonic} is not host-resolvable"
        )
    return ControlFlow(
        predicate=None, fallthrough=length, target=length,
        relative_to_head=True,
    )


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
            # These named fetch results come from byte-valued RAM even when
            # their exact masked-read circuit has a very loose naive
            # interval. Preserve that semantic range for symbolic shapes.
            r = (
                (0.0, 255.0)
                if (d.name or "").startswith("fetched_operand_")
                else evx(d.expr)
            )
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
                          window_range: tuple[int, int] | None = None,
                          return_bounds: bool = False):
    """Stores are licensed when the folded write address provably stays
    below $2000 -- the console only ever watches registers at or above it,
    and dispatch additionally requires no watch range below $2000, so a
    licensed store can never swallow a watch_write pause.  window_range
    (pinned windows) additionally forbids overlap with the window's own
    code bytes: the fold pinned them, so a self-modifying store would
    execute stale semantics.  Every step must still be non-halting.
    Returns whether any step stores, or ``(has_stores, bounds)`` when
    ``return_bounds`` is requested."""
    has_stores = False
    store_lo, store_hi = float("inf"), float("-inf")
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
            store_lo, store_hi = min(store_lo, lo), max(store_hi, hi)
            if window_range is not None:
                w_lo, w_hi = window_range
                if hi >= w_lo and lo < w_hi:
                    raise RuntimeError(
                        f"{label} step {k}: store bound [{lo:.0f}, {hi:.0f}] "
                        "may modify the window's own pinned bytes"
                    )
        if const_root(fold_result, "new_halted", k, label) != 0:
            raise RuntimeError(f"{label} step {k}: may halt")
    if return_bounds:
        bounds = None if not has_stores else (int(store_lo), int(store_hi))
        return has_stores, bounds
    return has_stores


def schedule_and_build(graph):
    fp = graph.program_graph
    sched = asap_schedule(fp.input_tokens, fp.output_tokens,
                          program_graph=fp, log=logger.debug)
    return build_model(program_graph=fp, schedule=sched)


def sparse_forward_work(model, n_layers: int) -> int:
    return sum(
        int(
            np.count_nonzero(model.attn[li].in_proj_weight)
            + np.count_nonzero(model.attn[li].out_proj.weight)
            + np.count_nonzero(model.ff_in[li].weight)
            + np.count_nonzero(model.ff_out[li].weight)
        )
        for li in range(n_layers)
    )


def prefer_inlined(materialized, inlined, label: str) -> bool:
    materialized_model, materialized_info = materialized
    inlined_model, inlined_info = inlined
    materialized_work = sparse_forward_work(
        materialized_model, materialized_info.n_layers,
    )
    inlined_work = sparse_forward_work(inlined_model, inlined_info.n_layers)
    accepted = (
        inlined_work <= materialized_work
        and inlined_info.n_layers <= materialized_info.n_layers
    )
    logger.debug(
        "%s: linear inlining %s (%d/%d layers, %d/%d sparse terms)",
        label, "accepted" if accepted else "rejected",
        inlined_info.n_layers, materialized_info.n_layers,
        inlined_work, materialized_work,
    )
    return accepted


def build_opcode_experts(pg, opcodes, *, schedule: str = "asap",
                         milp_seconds: int = 900,
                         inline_linear: bool | None = None,
                         structural_reads: bool = True) -> list[dict]:
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

    folder = GraphFolder(pg, root_names=RUNTIME_READ_DIMS,
                         structural_reads=structural_reads)
    experts = []
    for byte in opcodes:
        label = f"expert {byte:02X}"

        def fold_candidate(inline, byte=byte, label=label):
            candidate = folder.fold(
                opcode_pins(byte, OPCODES), label=f"{byte:02X}",
                inline_linear=inline,
            )
            if candidate.insert_side_hazards:
                raise RuntimeError(
                    f"{label}: decode pins reach insert-side expressions of "
                    f"{candidate.insert_side_hazards}; shared attention state "
                    "would diverge"
                )
            built = schedule_and_build(candidate)
            return candidate, built

        if inline_linear is None:
            materialized_r, materialized = fold_candidate(False)
            inlined_r, inlined = fold_candidate(True)
            r, built = (
                (inlined_r, inlined)
                if prefer_inlined(
                    (materialized[0], materialized[3]),
                    (inlined[0], inlined[3]), label,
                )
                else (materialized_r, materialized)
            )
        else:
            r, built = fold_candidate(inline_linear)
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
            model, _tokens, tok_to_idx, info = built
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


def build_sequence_expert(pg, head: int, code: bytes, repeat: int, *,
                          inline_linear: bool | None = None,
                          structural_reads: bool = True,
                          assume_taken_steps=()) -> dict:
    """One sequence expert: a K-instruction window folded as ONE forward.

    The router guarantees pc == head, the window's bytes, no pending NMI,
    and a remaining step/cycle budget covering the whole window, so the
    fold pins all of it.  The license: every step non-halting with stores
    bounded below $2000 (check_writes_licensed), control flow
    straight-line inside the window (folded next_pc == the next step's
    pinned pc), and later reads proven outside watched device ranges.
    ``assume_taken_steps`` is for a router with an independently certified
    state guard: those branch steps are pinned taken during construction,
    while all other branch decisions remain live.
    """
    label = f"sequence @{head:04X}"
    pin_list = sequence_pins(head, code, OPCODES) * repeat
    n_steps = len(pin_list)
    ops = decode_ops(code) * repeat
    assume_taken_steps = tuple(sorted(set(int(k) for k in assume_taken_steps)))
    for k in assume_taken_steps:
        if k < 0 or k >= n_steps:
            raise ValueError(f"{label}: assumed-taken step {k} is out of range")
        if ops[k].operation not in BRANCH_OPERATIONS:
            raise ValueError(
                f"{label}: assumed-taken step {k} is {ops[k].mnemonic}, "
                "not a branch"
            )
        pin_list[k] = dict(pin_list[k])
        pin_list[k]["branch_taken"] = 1.0
    folder = GraphFolder(pg, root_names=SEQUENCE_ROOT_DIMS,
                         structural_reads=structural_reads)

    def fold_candidate(inline):
        return folder.fold_sequence(
            pin_list, SEQUENCE_FEEDBACK, label=f"seq_{head:04X}",
            inline_linear=inline, bit_feedback_until=last_a_bit_step(ops),
        )

    r = fold_candidate(False if inline_linear is None else inline_linear)
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

    execution = project_sequence_execution(
        r, n_steps, pinned_sequence_control(r, pin_list, ops),
    )
    materialized = schedule_and_build(execution)
    if inline_linear is None:
        inlined_r = fold_candidate(True)
        if inlined_r.insert_side_hazards:
            raise RuntimeError(
                f"{label}: pins reach insert-side expressions "
                f"of {inlined_r.insert_side_hazards}"
            )
        inlined_execution = project_sequence_execution(
            inlined_r, n_steps,
            pinned_sequence_control(inlined_r, pin_list, ops),
        )
        inlined = schedule_and_build(inlined_execution)
        if prefer_inlined(
            (materialized[0], materialized[3]),
            (inlined[0], inlined[3]), label,
        ):
            r, execution, materialized = inlined_r, inlined_execution, inlined
    model, _tokens, _tok_to_idx, info = materialized
    cycles = sum(op.cycles for op in ops)
    # Window bytes join the name: different banks may carry different
    # windows at one head, and each needs its own step_core symbol.
    return {
        "kind": "seq", "name": f"seq_{head:04X}_{bytes(code).hex()}",
        "head": head, "has_stores": has_stores,
        "window": bytes(code), "n_steps": n_steps, "cycles": cycles,
        "cycles_prefix": cycles - ops[-1].cycles, "model": model,
        "step_spec": sequence_step_spec(execution, info),
        "union_lookup_of": r.union_lookup_of,
        "assume_taken_steps": assume_taken_steps,
        "layers": info.n_layers, "_d": info.d_model,
    }


# Union read name -> the persist carrying its address, for licensing
# non-constant surviving reads by interval.
READ_ADDR_DIMS = {
    "mem_A": "addrA", "mem_B": "addrB", "mem_C": "addrC",
    "operand_mem": "ea",
}

# Width policy for trace windows.  inline_linear trades depth for row
# width; a symbolic PLP byte fans its extracted flags into every
# downstream consumer and can blow one layer to ~10K terms (measured
# ~10x slower than the whole rest of the forward at 32 lanes).  Oversized
# inline builds fall back to the materialized fold; still-oversized
# windows are rejected outright.
INLINE_WORST_NNZ = 1200
MATERIALIZED_WORST_NNZ = 2600


def trace_read_bounds(k: int, op, body: bytes, operand_pinned: bool) -> dict:
    """Architecture-certified address bounds for step k's memory reads,
    keyed (step, union read name).  Sound by the 6502's addressing modes
    under a pinned opcode: zp,X/zp,Y and indirect pointer reads wrap in
    page zero; abs,X/abs,Y add at most 255 to a PINNED base; pulls stay in
    page one.  Used to prove non-constant reads disjoint from pending
    in-window stores, and to license their address range."""
    bounds: dict[tuple[int, str], tuple[int, int]] = {}
    if op.mode in ("zpx", "zpy"):
        bounds[(k, "operand_mem")] = (0x00, 0xFF)
    elif op.mode in ("abx", "aby") and operand_pinned:
        base = body[1] | (body[2] << 8)
        if base + 0xFF <= 0xFFFF:
            bounds[(k, "operand_mem")] = (base, base + 0xFF)
    elif op.mode in ("idx", "idy"):
        bounds[(k, "mem_A")] = (0x00, 0xFF)
        bounds[(k, "mem_B")] = (0x00, 0xFF)
    if op.operation in ("pla", "plp", "rts", "rti"):
        for nm in ("mem_A", "mem_B", "mem_C"):
            bounds[(k, nm)] = (0x100, 0x1FF)
    return bounds


def worst_layer_nnz(model, n_layers: int) -> int:
    return max(
        int(
            np.count_nonzero(model.attn[li].in_proj_weight)
            + np.count_nonzero(model.attn[li].out_proj.weight)
            + np.count_nonzero(model.ff_in[li].weight)
            + np.count_nonzero(model.ff_out[li].weight)
        )
        for li in range(n_layers)
    )


def build_trace_window_expert(pg, steps, *, s_pin: int | None = None,
                              symbolic=frozenset(),
                              inline_linear: bool = False,
                              structural_reads: bool = False,
                              flag_pins=None) -> dict:
    """One flow-following window: K profiled instructions as ONE forward.

    ``steps`` is [(pc, instruction bytes)] along the executed control flow
    -- JSR/RTS/JMP transfers included, so the window need not be
    contiguous.  Every executed opcode byte is pinned and must be
    dispatch-checked; bytes at addresses in ``symbolic`` (the profile saw
    them written) stay live and are read through the memory structure,
    with store-to-load forwarding resolving in-window writers.  ``s_pin``
    pins the entering stack pointer -- dispatch must guard S == s_pin --
    which folds every stack push/pull address to a constant the
    forwarding license can decide.  ``flag_pins`` pins entering P bits
    the profile saw constant at this head (state-input name -> 0/1);
    dispatch must guard the live P against exactly those bits, and the
    fold's feedback propagates them (a D pin of 0 deletes the decimal
    ADC/SBC circuit and its operand bit extraction from every step).

    Licenses, all checked: pins reach no insert-side expressions; every
    interior next_pc folds to the next step's pinned pc (JSR targets are
    static, RTS return addresses fold through forwarded pushes); no step
    halts; every store has a constant address below $2000 that overlaps
    no pinned byte; every surviving memory read is forwarded, at a
    constant RAM/ROM address, or interval-bounded inside RAM/ROM.
    """
    head = steps[0][0]
    label = f"trace @{head:04X}"
    pinned = pin_region_bytes(steps, symbolic, label)
    ops, pin_list, read_bounds = region_pin_list(
        steps, pinned, label, s_pin, flag_pins,
    )

    folder = GraphFolder(pg, root_names=RUNTIME_READ_DIMS,
                         structural_reads=structural_reads)

    def fold_and_license(inline: bool):
        r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK,
                                 label=f"trace_{head:04X}",
                                 inline_linear=inline,
                                 forward_stores=True,
                                 read_bounds=read_bounds,
                                 bit_feedback_until=last_a_bit_step(ops))
        spec = license_trace_fold(r, steps, pinned, label, read_bounds)
        return r, spec, *schedule_and_build(r)

    # Width ladder: full inlining (depth-optimal), then the materialized
    # fold.  A term-budgeted middle rung (inline_budget=16: inline depth,
    # materialized width) measured SLOWER than the deep/narrow fold at
    # 128 threads and was removed -- wide layers are what this mapping
    # pays for once threads cover the row count.
    r, spec_next, model, _tokens, tok_to_idx, info = fold_and_license(
        inline_linear
    )
    if inline_linear and worst_layer_nnz(model, info.n_layers) > INLINE_WORST_NNZ:
        r, spec_next, model, _tokens, tok_to_idx, info = fold_and_license(
            False
        )
    if worst_layer_nnz(model, info.n_layers) > MATERIALIZED_WORST_NNZ:
        raise RuntimeError(
            f"{label}: worst layer exceeds the width budget "
            f"({worst_layer_nnz(model, info.n_layers)} nnz)"
        )
    cycles = sum(op.cycles for op in ops)
    digest = hashlib.sha1(
        repr((tuple(steps), s_pin, sorted((flag_pins or {}).items()))).encode()
    ).hexdigest()[:8]
    return {
        "kind": "trace", "name": f"trace_{head:04X}_{digest}", "head": head,
        "n_steps": len(steps), "cycles": cycles,
        "cycles_prefix": cycles - ops[-1].cycles,
        "checks": sorted((a - head, b) for a, b in pinned.items()),
        "span": (min(pinned), max(pinned) + 1),
        "s_pin": s_pin, "flag_pins": dict(flag_pins or {}),
        "spec_next": spec_next, "model": model,
        "step_spec": fold_step_spec(r, model, tok_to_idx, info,
                                    n_steps=len(steps)),
        "union_lookup_of": r.union_lookup_of,
        "layers": info.n_layers, "_d": info.d_model,
    }


def license_trace_fold(
    r,
    steps,
    pinned,
    label,
    read_bounds,
) -> list[int]:
    """The trace-window licenses over one fold result (see
    build_trace_window_expert).  Returns spec_next: for every step whose
    next_pc could not be proven constant (a speculated branch), the pc the
    trace assumed comes next -- the dispatching kernel MUST verify the
    step's next_pc readout against it and commit only the prefix before
    the first violation; -1 marks proven or final steps."""
    if r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions "
            f"of {r.insert_side_hazards}"
        )
    n_steps = len(steps)
    spec_next = [-1] * n_steps
    for k in range(n_steps - 1):
        try:
            got = const_root(r, "next_pc", k, label)
        except RuntimeError:
            got = None
        if got is None:
            if OPCODES[steps[k][1][0]].operation not in BRANCH_OPERATIONS:
                raise SequenceRejection(
                    f"{label} s{k}: non-branch control flow is not constant",
                    step=k,
                )
            spec_next[k] = steps[k + 1][0]
        elif got != steps[k + 1][0]:
            raise SequenceRejection(
                f"{label} s{k}: control flow deviates from the trace",
                step=k,
            )
    license_step_effects(r, n_steps, pinned, label, read_bounds)
    return spec_next


def license_step_effects(
    r,
    n_steps: int,
    pinned,
    label,
    read_bounds,
) -> None:
    """The per-step memory/halt licenses shared by trace windows and
    predicated regions: no step halts, every store has a constant address
    below $2000 that overlaps no pinned byte, and every surviving read is
    forwarded, at a constant RAM/ROM address, or interval-bounded inside
    RAM/ROM.  Every check is per NODE, so a region's arms are licensed
    exactly as if each were its own straight-line window."""
    for k in range(n_steps):
        if const_root(r, "new_halted", k, label) != 0:
            raise SequenceRejection(f"{label} s{k}: may halt", step=k)
        for store, addr in STORE_SLOTS:
            if const_root(r, store, k, label) == 0:
                continue
            a = const_root(r, addr, k, label)
            if not 0 <= a < 0x2000:
                raise SequenceRejection(
                    f"{label} s{k}: store at {a:04X} is not below $2000",
                    step=k,
                )
            if a in pinned:
                raise SequenceRejection(
                    f"{label} s{k}: store hits pinned byte {a:04X}", step=k,
                )
    for k, name, a, forwarded in r.read_log:
        if forwarded:
            continue
        if a is not None:
            if a < 0x2000 or a >= 0x8000:
                continue
            raise SequenceRejection(
                f"{label} s{k}: read {name} at {a:04X} may reach a device",
                step=k,
            )
        b = read_bounds.get((k, name))
        if b is not None and (b[1] < 0x2000 or b[0] >= 0x8000):
            continue
        addr_dim = READ_ADDR_DIMS.get(name)
        dim = r.dim_of_name.get(f"{addr_dim}__s{k}") if addr_dim else None
        if dim is None:
            raise SequenceRejection(
                f"{label} s{k}: unbounded read {name}", step=k,
            )
        lo, hi = interval_of(r, dim)
        if not (hi < 0x2000 or lo >= 0x8000):
            raise SequenceRejection(
                f"{label} s{k}: read {name} bound [{lo:.0f}, {hi:.0f}] "
                "may reach a device", step=k,
            )


def region_pin_list(nodes, pinned, label, s_pin, flag_pins):
    """Per-node pins plus certified read bounds for a region/trace tree:
    every node's pc, opcode and resident operand bytes pin; the ROOT
    additionally pins the guarded entering S and P bits."""
    ops, pin_list = [], []
    read_bounds: dict[tuple[int, str], tuple[int, int]] = {}
    for k, (pc, body) in enumerate(nodes):
        op = OPCODES.get(body[0])
        if op is None or len(body) != op.length:
            raise RuntimeError(f"{label} s{k}: bad instruction at {pc:04X}")
        if pc not in pinned:
            raise RuntimeError(f"{label} s{k}: opcode byte is symbolic")
        ops.append(op)
        pins = opcode_pins(op.byte, OPCODES)
        pins["current_pc_lo"] = float(pc & 0xFF)
        pins["current_pc_hi"] = float((pc >> 8) & 0xFF)
        pins["current_halted"] = 0.0
        lo, hi = (pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF
        if lo in pinned:
            pins["fetched_operand_lo_all"] = float(pinned[lo])
        if hi in pinned:
            pins["fetched_operand_hi_all"] = float(pinned[hi])
        if k == 0 and s_pin is not None:
            pins["current_s"] = float(s_pin)
        if k == 0 and flag_pins:
            pins.update({n: float(v) for n, v in flag_pins.items()})
        pin_list.append(pins)
        operand_pinned = all((pc + j) in pinned for j in range(1, op.length))
        read_bounds.update(trace_read_bounds(k, op, body, operand_pinned))
    return ops, pin_list, read_bounds


def pin_region_bytes(nodes, symbolic, label) -> dict[int, int]:
    """Address -> byte for every resident byte a region's nodes execute.
    Two nodes may share an address (arms rejoin) as long as they agree."""
    pinned: dict[int, int] = {}
    for pc, body in nodes:
        for j, byte in enumerate(body):
            addr = (pc + j) & 0xFFFF
            if addr in symbolic:
                continue
            if pinned.setdefault(addr, byte) != byte:
                raise RuntimeError(
                    f"{label}: byte {addr:04X} pinned twice with different "
                    "values"
                )
    return pinned


def build_shape_experts(pg, shapes, *,
                        inline_linear: bool | None = None,
                        structural_reads: bool = True,
                        allow_stores: bool = False,
                        allow_runtime_store_guard: bool = False,
                        ) -> tuple[list[dict], list[dict]]:
    """Canonical shape experts for the runtime window table.

    Each shape folds ONCE with pc and every operand byte symbolic
    (fold_sequence pc_offsets); the license: only the first instruction may
    read memory, every step non-halting with stores bounded below $2000
    (a symbolic zero-page operand bounds itself), and each interior
    step's next_pc equals head + offset -- verified EXHAUSTIVELY over every
    head the window fits at, which is possible exactly because the folded
    expression may depend on nothing but the pc bytes.  The JMP self-spin's
    closure is a dispatch property instead (the router compares the operand
    bytes to the head). Stores remain rejected by default. Opt-in routers
    may accept statically bounded stores, or one structurally certified
    direct absolute store when they enforce the returned live-address
    guard. A library shape that fails its selected license is a hard error:
    the library is code.
    """
    folder = GraphFolder(pg, root_names=SEQUENCE_ROOT_DIMS,
                         structural_reads=structural_reads)
    experts: list[dict] = []
    meta: list[dict] = []
    for sid, shape in enumerate(shapes):
        name, template, repeat = shape[:3]
        speculate_branches = bool(shape[3]) if len(shape) > 3 else False
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
        pins, offsets = sequence_shape_pins(template, OPCODES)
        pin_list = pins * repeat
        pc_offsets = offsets * repeat
        n_steps = len(pin_list)
        def fold_candidate(
            inline, pin_list=pin_list, name=name, pc_offsets=pc_offsets,
            ops=ops, repeat=repeat,
        ):
            return folder.fold_sequence(
                pin_list, SEQUENCE_FEEDBACK, label=f"shape_{name}",
                pc_offsets=pc_offsets, inline_linear=inline,
                bit_feedback_until=last_a_bit_step(ops * repeat),
            )

        r = fold_candidate(False if inline_linear is None else inline_linear)
        if r.insert_side_hazards:
            raise RuntimeError(
                f"{label}: pins reach insert-side expressions "
                f"of {r.insert_side_hazards}"
            )
        check_reads_licensed(r, ops * repeat, label)
        store_guard_offset = None
        direct_abs = [
            (k, offs[k] + 1)
            for k, op in enumerate(ops)
            if op.operation in {"store_a", "store_x", "store_y"}
            and op.mode == "abs"
        ]
        try:
            has_stores, store_bounds = check_writes_licensed(
                r, n_steps, label, return_bounds=True,
            )
        except RuntimeError as exc:
            if not allow_runtime_store_guard or len(direct_abs) != 1:
                raise
            store_step, store_guard_offset = direct_abs[0]
            # Narrow structural certificate: one ordinary absolute store,
            # with no other write roots. Dispatch will bound and separate
            # its live operand-derived address.
            for k in range(n_steps):
                expected = int(k == store_step)
                if const_root(r, "is_store", k, label) != expected:
                    raise RuntimeError(
                        f"{label}: runtime store guard does not cover step {k}"
                    ) from exc
                if (
                    const_root(r, "is_store2", k, label)
                    or const_root(r, "is_store3", k, label)
                    or const_root(r, "new_halted", k, label)
                ):
                    raise RuntimeError(
                        f"{label}: runtime store guard has extra writes or halt"
                    ) from exc
            has_stores, store_bounds = True, None
        if has_stores and not allow_stores:
            # A shape's head is symbolic, so store disjointness from the
            # window's own bytes cannot be proven; store idioms stay
            # pinned-window-only until a head-range dispatch guard exists.
            raise RuntimeError(f"{label}: store shapes are not licensed")
        spec_offsets = [-1] * n_steps
        if not self_target:
            for k in range(n_steps - 1):
                expected = pc_offsets[k + 1]
                try:
                    heads, got = eval_over_heads(
                        r, r.dim_of_name[f"next_pc__s{k}"], len(template)
                    )
                    proven = np.array_equal(got, heads + expected)
                except RuntimeError:
                    proven = False
                if (
                    not proven
                    and speculate_branches
                    and ops[k].operation in BRANCH_OPERATIONS
                ):
                    spec_offsets[k] = expected
                    continue
                if not proven:
                    raise RuntimeError(
                        f"{label} step {k}: interior next_pc is not "
                        "head + offset for every head"
                    )

        execution = project_sequence_execution(
            r, n_steps,
            shape_sequence_control(
                r, n_steps, ops, offs, len(template), self_target,
            ),
        )
        materialized = schedule_and_build(execution)
        if inline_linear is None:
            inlined_r = fold_candidate(True)
            if inlined_r.insert_side_hazards:
                raise RuntimeError(
                    f"{label}: pins reach insert-side expressions "
                    f"of {inlined_r.insert_side_hazards}"
                )
            inlined_execution = project_sequence_execution(
                inlined_r, n_steps,
                shape_sequence_control(
                    inlined_r, n_steps, ops, offs, len(template), self_target,
                ),
            )
            inlined = schedule_and_build(inlined_execution)
            if prefer_inlined(
                (materialized[0], materialized[3]),
                (inlined[0], inlined[3]), label,
            ):
                r, execution, materialized = inlined_r, inlined_execution, inlined
        model, _tokens, _tok_to_idx, info = materialized
        ops_full = ops * repeat
        cycles = sum(op.cycles for op in ops_full)
        experts.append({
            "kind": "shape", "name": f"shape_{name}", "shape_id": sid,
            "opcode_checks": checks, "length": len(template),
            "self_target": self_target, "n_steps": n_steps,
            "has_stores": has_stores, "store_bounds": store_bounds,
            "store_guard_offset": store_guard_offset,
            "spec_offsets": spec_offsets,
            "cycles": cycles,
            "cycles_prefix": cycles - ops_full[-1].cycles, "model": model,
            "step_spec": sequence_step_spec(execution, info),
            "union_lookup_of": r.union_lookup_of,
            "layers": info.n_layers, "_d": info.d_model,
        })
        meta.append({
            "id": sid, "name": name, "sig": [[o, b] for o, b in checks],
            "length": len(template), "repeat": repeat,
            "close": (
                "jmp" if self_target
                else "branch" if ops[-1].operation in BRANCH_OPERATIONS
                else "none"
            ),
        })
    if experts:
        logger.info(
            "shape experts: %d folded, %d-%d layers, d_model %d-%d",
            len(experts),
            min(e["layers"] for e in experts), max(e["layers"] for e in experts),
            min(e["_d"] for e in experts), max(e["_d"] for e in experts),
        )
    return experts, meta
