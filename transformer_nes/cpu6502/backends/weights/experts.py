"""Fold-and-build expert models for the generated weights engine.

Each builder partially evaluates the union graph (graph/fold.py), checks its
dispatch license (checked, never assumed), and compiles the folded graph into
its own transformer.  The machine (weights_runner) owns orchestration and the
engine cache; this module owns folds, licenses, and model construction.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace

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
    RecurrenceSummary,
    StateField,
)
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
    "new_a", "new_x", "new_y", "new_s",
    "p_c", "p_z", "p_i", "p_d", "p_b", "p_v", "p_n",
)

SEQUENCE_FLAG_DIMS = (
    ("p_c", "new_p_c_final", reference.FLAG_C, "in_p_c"),
    ("p_z", "new_p_z_final", reference.FLAG_Z, "in_p_z"),
    ("p_i", "new_p_i_final", reference.FLAG_I, "in_p_i"),
    ("p_d", "new_p_d_final", reference.FLAG_D, "in_p_d"),
    ("p_b", "new_p_b_final", reference.FLAG_B, "in_p_b"),
    ("p_v", "new_p_v_final", reference.FLAG_V, "in_p_v"),
    ("p_n", "new_p_n_final", reference.FLAG_N, "in_p_n"),
)

SEQUENCE_PRESERVE_INPUTS = {
    "new_a": (("in_a", 1),),
    "new_x": (("in_x", 1),),
    "new_y": (("in_y", 1),),
    "new_s": (("in_s", 1),),
    **{name: ((input_name, 1),)
       for name, _dim_name, _mask, input_name in SEQUENCE_FLAG_DIMS},
}

WRITE_RECEIPTS = (
    ("is_store", "write_addr_out", "write_value_out"),
    ("is_store2", "write_addr2_out", "write_value2_out"),
    ("is_store3", "write_addr3_out", "write_value3_out"),
)
SEQUENCE_ROOT_DIMS = (
    "new_a", "new_x", "new_y", "new_s",
    *(dim_name for _name, dim_name, _mask, _input_name in SEQUENCE_FLAG_DIMS),
    "ea", "next_pc", "new_halted", "branch_taken",
    *(name for receipt in WRITE_RECEIPTS for name in receipt),
)


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
    recurrence: RecurrenceSummary | None = None,
) -> ExecutionGraph:
    """Replace the proof graph's per-step trace ABI with its execution ABI.

    The complete fold is retained until all control, read, write and halt
    licenses have passed.  Only then do we keep the final architectural
    state and the address/value of each statically-live ordered write.  The
    projected graph is what gets scheduled and converted to weights.
    """
    final_names = {
        name: f"{name}__s{n_steps - 1}"
        for name in SEQUENCE_STATE_DIMS
        if not name.startswith("p_")
    }
    final_names.update({
        name: f"{dim_name}__s{n_steps - 1}"
        for name, dim_name, _mask, _input_name in SEQUENCE_FLAG_DIMS
    })
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

    if control.predicate is not None and not control.flag_mask:
        output_tokens["control:branch"] = Expression({control.predicate: 1})
    execution_control = (
        replace(control, predicate=None) if control.flag_mask else control
    )

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
        control=execution_control,
        recurrence=recurrence,
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
        "flag_mask": control.flag_mask,
        "flag_value": control.flag_value,
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


BRANCH_FLAG = {
    "bcc": (reference.FLAG_C, False),
    "bcs": (reference.FLAG_C, True),
    "beq": (reference.FLAG_Z, True),
    "bmi": (reference.FLAG_N, True),
    "bne": (reference.FLAG_Z, False),
    "bpl": (reference.FLAG_N, False),
    "bvc": (reference.FLAG_V, False),
    "bvs": (reference.FLAG_V, True),
}


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
            flag_mask=BRANCH_FLAG[op.operation][0],
            flag_value=BRANCH_FLAG[op.operation][1],
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
            flag_mask=BRANCH_FLAG[op.operation][0],
            flag_value=BRANCH_FLAG[op.operation][1],
        )
    if op.operation in SEQUENCE_WRITERS or op.operation == "jmp":
        raise RuntimeError(
            f"shape final control {op.mnemonic} is not host-resolvable"
        )
    return ControlFlow(
        predicate=None, fallthrough=length, target=length,
        relative_to_head=True,
    )


def semantic_recurrence(
    ops, offsets, *, self_target: bool = False,
) -> RecurrenceSummary | None:
    if self_target:
        return RecurrenceSummary(
            kind="identity", steps_per_iteration=1,
            cycles_per_iteration=ops[0].cycles, cycles_prefix=0,
        )
    affine = {
        "dex": ("x", -1), "inx": ("x", 1),
        "dey": ("y", -1), "iny": ("y", 1),
    }
    if (
        len(ops) == 2
        and ops[0].operation in affine
        and ops[1].operation == "bne"
    ):
        register, delta = affine[ops[0].operation]
        length = sum(op.length for op in ops)
        return RecurrenceSummary(
            kind="affine_until",
            steps_per_iteration=2,
            cycles_per_iteration=sum(op.cycles for op in ops),
            cycles_prefix=sum(op.cycles for op in ops[:-1]),
            register=register,
            delta=delta,
            fallthrough_offset=length,
            closure_operand_offset=offsets[1] + 1,
            closure_operand_value=(-length) & 0xFF,
        )
    prefix = tuple(op.operation for op in ops[:-1])
    poll_prefixes = {
        ("load_a",), ("bit",), ("load_a", "and"),
    }
    if (
        len(ops) >= 2
        and prefix in poll_prefixes
        and ops[-1].operation in BRANCH_OPERATIONS
        and ops[0].operation in READS_MEMORY
        and ops[0].mode not in ("imp", "acc", "imm")
    ):
        length = sum(op.length for op in ops)
        return RecurrenceSummary(
            kind="invariant_poll",
            steps_per_iteration=len(ops),
            cycles_per_iteration=sum(op.cycles for op in ops),
            cycles_prefix=sum(op.cycles for op in ops[:-1]),
            fallthrough_offset=length,
            closure_operand_offset=offsets[-1] + 1,
            closure_operand_value=(-length) & 0xFF,
        )
    return None


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


def build_opcode_experts(pg, opcodes) -> list[dict]:
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
                         structural_reads=True)
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


def opcode_execution_bytes(opcodes) -> tuple[int, ...]:
    eligible = []
    for byte in opcodes or ():
        byte = int(byte)
        op = OPCODES.get(byte)
        if op is None:
            continue
        writes_memory = (
            op.operation in MEMORY_WRITE_OPERATIONS and op.mode != "acc"
        )
        if (
            writes_memory
            or op.operation in SEQUENCE_WRITERS
            or op.operation in {"jmp", "pha", "php"}
        ):
            continue
        eligible.append(byte)
    return tuple(eligible)


def build_opcode_execution_experts(pg, opcodes) -> list[dict]:
    eligible = [
        (byte, OPCODES[byte]) for byte in opcode_execution_bytes(opcodes)
    ]
    if not eligible:
        return []

    shapes = [
        (f"exec_{byte:02X}", bytes([byte] + [0] * (op.length - 1)), 1)
        for byte, op in eligible
    ]
    experts, _meta = build_shape_experts(pg, shapes)
    for expert, (byte, _op) in zip(experts, eligible, strict=True):
        if expert["recurrence"] is not None:
            raise RuntimeError(
                f"compact opcode {byte:02X} escaped its execution license"
            )
        expert.update(kind="exec_opcode", byte=byte)
    logger.info(
        "compact execution experts: %d opcodes, %d layers, %d sparse terms",
        len(experts),
        sum(expert["layers"] for expert in experts),
        sum(sparse_forward_work(expert["model"], expert["layers"])
            for expert in experts),
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
    base_ops = decode_ops(code)
    ops = base_ops * repeat
    base_offsets = [0]
    for op in base_ops[:-1]:
        base_offsets.append(base_offsets[-1] + op.length)
    folder = GraphFolder(pg, root_names=SEQUENCE_ROOT_DIMS,
                         structural_reads=True)

    def fold_candidate(inline):
        return folder.fold_execution_sequence(
            pin_list, SEQUENCE_FEEDBACK, label=f"seq_{head:04X}",
            inline_linear=inline, bit_feedback_until=last_a_bit_step(ops),
        )

    r = fold_candidate(False)
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

    def execution_control(candidate):
        control = pinned_sequence_control(candidate, pin_list, ops)
        self_target = (
            len(base_ops) == 1
            and base_ops[0].operation == "jmp"
            and base_ops[0].mode == "abs"
            and control.target == head
        )
        recurrence = semantic_recurrence(
            base_ops, base_offsets, self_target=self_target,
        )
        if recurrence is not None and recurrence.kind != "identity" and (
            control.predicate is None or control.target != head
        ):
            recurrence = None
        return control, recurrence

    control, recurrence = execution_control(r)
    execution = project_sequence_execution(
        r, n_steps, control, recurrence,
    )
    materialized = schedule_and_build(execution)
    inlined_r = fold_candidate(True)
    if inlined_r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions "
            f"of {inlined_r.insert_side_hazards}"
        )
    inlined_control, inlined_recurrence = execution_control(inlined_r)
    inlined_execution = project_sequence_execution(
        inlined_r, n_steps, inlined_control, inlined_recurrence,
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
        "recurrence": execution.recurrence,
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
    bytes to the head). Stores and speculative interior control are rejected;
    a library shape that fails its license is a hard error because the library
    is code.
    """
    folder = GraphFolder(pg, root_names=SEQUENCE_ROOT_DIMS,
                         structural_reads=True)
    experts: list[dict] = []
    meta: list[dict] = []
    for sid, shape in enumerate(shapes):
        name, template, repeat = shape
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
            return folder.fold_execution_sequence(
                pin_list, SEQUENCE_FEEDBACK, label=f"shape_{name}",
                pc_offsets=pc_offsets, inline_linear=inline,
                bit_feedback_until=last_a_bit_step(ops * repeat),
            )

        r = fold_candidate(False)
        if r.insert_side_hazards:
            raise RuntimeError(
                f"{label}: pins reach insert-side expressions "
                f"of {r.insert_side_hazards}"
            )
        check_reads_licensed(r, ops * repeat, label)
        if check_writes_licensed(r, n_steps, label):
            # A shape's head is symbolic, so store disjointness from the
            # window's own bytes cannot be proven. Store idioms belong to
            # concrete sequence modules, where that proof is static.
            raise RuntimeError(f"{label}: store shapes are not licensed")
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
                if not proven:
                    raise RuntimeError(
                        f"{label} step {k}: interior next_pc is not "
                        "head + offset for every head"
                    )

        recurrence = semantic_recurrence(ops, offs, self_target=self_target)
        execution = project_sequence_execution(
            r, n_steps,
            shape_sequence_control(
                r, n_steps, ops, offs, len(template), self_target,
            ),
            recurrence,
        )
        materialized = schedule_and_build(execution)
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
            recurrence,
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
            "cycles": cycles,
            "cycles_prefix": cycles - ops_full[-1].cycles, "model": model,
            "step_spec": sequence_step_spec(execution, info),
            "recurrence": execution.recurrence,
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
