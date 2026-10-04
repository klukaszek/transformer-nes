"""Construct the fixed opcode, interrupt and shape expert bank.

The stages are explicit: fold a candidate, prove its dispatch license,
project architectural effects, schedule weights, then lower its runtime ABI.
Execution contracts live in execution.py; numerical licenses live in proofs.py.
"""

from __future__ import annotations

import functools
import logging
from typing import Literal

import numpy as np

from transformer_nes.cpu6502.backends.weights.execution import (
    SEQUENCE_ROOT_DIMS,
    WRITE_RECEIPTS,
    ControlFlow,
    lower_execution,
    opcode_execution_control,
    project_sequence_execution,
    shape_sequence_control,
    shape_timing,
)
from transformer_nes.cpu6502.backends.weights.folds import (
    SEQUENCE_FEEDBACK,
    SequenceFolder,
    sequence_shape_pins,
)
from transformer_nes.cpu6502.backends.weights.proofs import (
    check_reads_licensed,
    const_root,
    eval_over_heads,
)
from transformer_nes.cpu6502.backends.weights.shapes import SHAPE_LIBRARY, last_a_bit_step
from transformer_nes.cpu6502.spec import BRANCH_OPERATIONS, OPCODES
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.weights import build_model

logger = logging.getLogger(__name__)


def schedule_and_build(graph):
    fp = graph.program_graph
    sched = asap_schedule(fp, log=logger.debug)
    return build_model(program_graph=fp, schedule=sched, recycle_slots=False)


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
        materialized_model,
        materialized_info.n_layers,
    )
    inlined_work = sparse_forward_work(inlined_model, inlined_info.n_layers)
    accepted = (
        inlined_work <= materialized_work and inlined_info.n_layers <= materialized_info.n_layers
    )
    logger.debug(
        "%s: linear inlining %s (%d/%d layers, %d/%d sparse terms)",
        label,
        "accepted" if accepted else "rejected",
        inlined_info.n_layers,
        materialized_info.n_layers,
        inlined_work,
        materialized_work,
    )
    return accepted


def build_opcode_execution_experts(pg, opcodes) -> list[dict]:
    eligible = [(int(byte), OPCODES[int(byte)]) for byte in opcodes or () if int(byte) in OPCODES]
    if not eligible:
        return []

    shapes = [
        (f"exec_{byte:02X}", bytes([byte] + [0] * (op.length - 1)), 1) for byte, op in eligible
    ]
    experts, _meta = _build_symbolic_execution_experts(
        pg,
        shapes,
        expert_kind="exec_opcode",
    )
    for expert, (byte, _op) in zip(experts, eligible, strict=True):
        if expert["spin"]:
            raise RuntimeError(f"compact opcode {byte:02X} escaped its execution license")
        expert["byte"] = byte
    return experts


def build_interrupt_experts(pg) -> list[dict]:
    """Build folded NMI and IRQ experts."""
    experts: list[dict] = []
    for name, pins in (
        ("int_nmi", {"is_nmi": 1.0, "is_irq": 0.0}),
        ("int_irq", {"is_nmi": 0.0, "is_irq": 1.0}),
    ):
        label = f"interrupt {name}"
        folder = sequence_folder(pg)
        r = folder.fold_execution_sequence(
            [pins],
            SEQUENCE_FEEDBACK,
            label=name,
            pc_offsets=[0],
            inline_linear=False,
        )
        if r.insert_side_hazards:
            raise RuntimeError(
                f"{label}: pins reach insert-side expressions of {r.insert_side_hazards}"
            )
        control = ControlFlow(
            predicate=None,
            fallthrough=0,
            target=0,
            target_dimension=r.dim_of_name["next_pc__s0"],
        )
        execution = project_sequence_execution(r, 1, control)
        model, _tok_to_idx, info = schedule_and_build(execution)
        experts.append(
            {
                "kind": "exec_int",
                "name": name,
                "nmi": name == "int_nmi",
                "n_steps": 1,
                "model": model,
                "step_spec": lower_execution(execution, info),
                "union_lookup_of": r.union_lookup_of,
            }
        )
    return experts


@functools.cache
def sequence_folder(pg) -> SequenceFolder:
    """The folder of `pg`'s execution folds (one per graph: it caches)."""
    return SequenceFolder(pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True)


def _build_execution_expert(
    folder: SequenceFolder,
    shape: tuple[str, bytes, int],
    sid: int,
    expert_kind: Literal["shape", "exec_opcode"],
) -> tuple[dict, dict]:
    """Fold and license one template before projecting or emitting its effects."""
    name, template, repeat = shape
    label = f"shape {name}"
    pins, offsets = sequence_shape_pins(template, OPCODES)
    ops = [OPCODES[template[offset]] for offset in offsets]
    checks = list(zip(offsets, (op.byte for op in ops), strict=True))
    self_target = (
        expert_kind == "shape"
        and len(ops) == 1
        and ops[0].operation == "jmp"
        and ops[0].mode == "abs"
    )
    if repeat > 1 and not self_target:
        raise RuntimeError(f"{label}: only a JMP self-spin may repeat")
    pin_list = pins * repeat
    pc_offsets = offsets * repeat
    n_steps = len(pin_list)

    def fold_candidate(inline):
        return folder.fold_execution_sequence(
            pin_list,
            SEQUENCE_FEEDBACK,
            label=f"shape_{name}",
            pc_offsets=pc_offsets,
            inline_linear=inline,
            bit_feedback_until=last_a_bit_step(ops * repeat),
        )

    r = fold_candidate(False)
    if r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions of {r.insert_side_hazards}"
        )
    check_reads_licensed(r, ops * repeat, label)
    if expert_kind == "exec_opcode":
        if n_steps != 1:
            raise RuntimeError("staged effects require one instruction")
        if ops[0].operation != "brk" and const_root(r, "new_halted", 0, label) != 0:
            raise RuntimeError(f"{label}: may halt")
    elif any(
        const_root(r, store, k, label)
        for k in range(n_steps)
        for store, _addr, _value in WRITE_RECEIPTS
    ):
        # Fixed shapes cannot hide effects from the device pause protocol.
        raise RuntimeError(f"{label}: store shapes are not licensed")
    if not self_target:
        for k in range(n_steps - 1):
            expected = pc_offsets[k + 1]
            try:
                heads, got = eval_over_heads(r, r.dim_of_name[f"next_pc__s{k}"], len(template))
                proven = np.array_equal(got, heads + expected)
            except RuntimeError:
                proven = False
            if not proven:
                raise RuntimeError(
                    f"{label} step {k}: interior next_pc is not head + offset for every head"
                )

    def execution_of(candidate):
        control = (
            opcode_execution_control(candidate, ops[0])
            if expert_kind == "exec_opcode"
            else shape_sequence_control(
                candidate,
                n_steps,
                ops,
                offsets,
                len(template),
                self_target,
            )
        )
        halt = (
            candidate.dim_of_name["new_halted__s0"]
            if expert_kind == "exec_opcode" and ops[0].operation == "brk"
            else None
        )
        return project_sequence_execution(
            candidate,
            n_steps,
            control,
            halt,
        )

    execution = execution_of(r)
    materialized = schedule_and_build(execution)
    inlined_r = fold_candidate(True)
    if inlined_r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions of {inlined_r.insert_side_hazards}"
        )
    inlined_execution = execution_of(inlined_r)
    inlined = schedule_and_build(inlined_execution)
    if prefer_inlined(
        (materialized[0], materialized[2]),
        (inlined[0], inlined[2]),
        label,
    ):
        r, execution, materialized = inlined_r, inlined_execution, inlined
    model, _tok_to_idx, info = materialized
    expert = {
        "kind": expert_kind,
        "name": f"shape_{name}",
        "shape_id": sid,
        "opcode_checks": checks,
        "length": len(template),
        "n_steps": n_steps,
        "model": model,
        "step_spec": lower_execution(execution, info),
        **shape_timing(execution, self_target),
        "union_lookup_of": r.union_lookup_of,
    }
    metadata = {
        "id": sid,
        "name": name,
        "sig": [[o, b] for o, b in checks],
        "length": len(template),
        "repeat": repeat,
        "close": (
            "jmp" if self_target else "branch" if ops[-1].operation in BRANCH_OPERATIONS else "none"
        ),
    }
    return expert, metadata


def _build_symbolic_execution_experts(
    pg,
    shapes,
    *,
    expert_kind: Literal["shape", "exec_opcode"],
) -> tuple[list[dict], list[dict]]:
    """Build licensed templates in their stable dispatch order."""
    folder = sequence_folder(pg)
    experts, metadata = [], []
    for sid, shape in enumerate(shapes):
        expert, record = _build_execution_expert(folder, shape, sid, expert_kind)
        experts.append(expert)
        metadata.append(record)
    return experts, metadata


def build_shape_experts(pg, shapes) -> tuple[list[dict], list[dict]]:
    return _build_symbolic_execution_experts(
        pg,
        shapes,
        expert_kind="shape",
    )


def build_core_experts(pg) -> tuple[list[dict], list[dict]]:
    """The core engine's experts in its build order (the forward hook's ids
    0..): per opcode, the interrupts, the canonical shapes; and the shapes'
    metadata."""
    experts = build_opcode_execution_experts(pg, sorted(OPCODES))
    experts.extend(build_interrupt_experts(pg))
    shape_experts, shapes_meta = build_shape_experts(pg, SHAPE_LIBRARY)
    return experts + shape_experts, shapes_meta
