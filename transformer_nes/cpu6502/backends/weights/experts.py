"""Construct the fixed opcode, interrupt and shape expert bank.

The stages are explicit: fold a candidate, prove its dispatch license,
project architectural effects, schedule weights, then lower its runtime ABI.
Execution contracts live in execution.py; numerical licenses live in proofs.py.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from transformer_nes.cpu6502.backends.weights.execution import (
    SEQUENCE_ROOT_DIMS,
    STATE_FIELDS,
    WRITE_RECEIPTS,
    ArchitecturalOutput,
    ExecutionGraph,
    lower_execution,
    project_sequence_execution,
    readout_slots,
    shape_timing,
    step_spec_base,
)
from transformer_nes.cpu6502.backends.weights.folds import (
    SEQUENCE_FEEDBACK,
    SequenceFolder,
    opcode_pins,
    sequence_shape_pins,
)
from transformer_nes.cpu6502.backends.weights.proofs import (
    check_reads_licensed,
    const_root,
    eval_over_heads,
)
from transformer_nes.cpu6502.backends.weights.shapes import (
    COUNTERS,
    SHAPE_LIBRARY,
    counter_recurrence_graph,
    last_a_bit_step,
)
from transformer_nes.cpu6502.spec import BRANCH_OPERATIONS, MEMORY_MODES, OPCODES, READS_MEMORY
from transformer_nes.graph.core import MemoryReadDimension
from transformer_nes.graph.fold import GraphFolder, const_of, linear_input_form
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.torch_layout import layer_arrays
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


def build_opcode_execution_experts(pg, opcodes, *, resume_reads=False) -> list[dict]:
    """Construct whole instruction checkpoints or production continuations.

    Both use the same graph; only production binds a read prefix's packet.
    """
    eligible = [(int(byte), OPCODES[int(byte)]) for byte in opcodes or () if int(byte) in OPCODES]
    if not eligible:
        return []

    shapes = [
        (f"exec_{byte:02X}", bytes([byte] + [0] * (op.length - 1)), 1) for byte, op in eligible
    ]
    experts, _meta = _build_symbolic_execution_experts(
        pg,
        shapes,
        expert_kind="exec_resume" if resume_reads else "exec_opcode",
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
        execution = project_sequence_execution(r, 1)
        model, _tok_to_idx, info = schedule_and_build(execution)
        spec = lower_execution(execution, info)
        experts.append(
            {
                "kind": "exec_int",
                "name": name,
                "nmi": name == "int_nmi",
                "n_steps": 1,
                "model": model,
                "step_spec": spec,
                "output_slots": readout_slots(spec),
            }
        )
    return experts


def sequence_folder(pg) -> SequenceFolder:
    """A folder sharing the evaluator owned weakly by the live source graph."""
    return SequenceFolder(pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True)


def _build_execution_expert(
    folder: SequenceFolder,
    shape: tuple[str, bytes, int],
    sid: int,
    expert_kind: Literal["shape", "exec_opcode", "exec_resume"],
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

    read_resume = (
        expert_kind == "exec_resume"
        and ops[0].operation in READS_MEMORY
        and ops[0].mode in MEMORY_MODES
    )
    bindings = None
    if read_resume:
        bindings = {"ea": {"in_read_addr_lo": 1, "in_read_addr_hi": 256}}
        timing = folder.evaluate(pin_list[0]).val.get(folder.by_name["cycles"])
        if timing is None or const_of(timing, folder.pg.one) is None:
            bindings["cycles"] = {"in_read_cycles": 1}

    def fold_candidate(inline):
        return folder.fold_execution_sequence(
            pin_list,
            SEQUENCE_FEEDBACK,
            label=f"shape_{name}",
            pc_offsets=pc_offsets,
            inline_linear=inline,
            bit_feedback_until=last_a_bit_step(ops * repeat),
            bindings=bindings,
        )

    r = fold_candidate(False)
    if r.insert_side_hazards:
        raise RuntimeError(
            f"{label}: pins reach insert-side expressions of {r.insert_side_hazards}"
        )
    check_reads_licensed(r, ops * repeat, label)
    if expert_kind in ("exec_opcode", "exec_resume"):
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
        halt = (
            candidate.dim_of_name["new_halted__s0"]
            if expert_kind in ("exec_opcode", "exec_resume") and ops[0].operation == "brk"
            else None
        )
        result = project_sequence_execution(candidate, n_steps, halt)
        if read_resume:
            for dim in result.program_graph.all_dims:
                if isinstance(dim, MemoryReadDimension):
                    form = linear_input_form(dim.memory_read.address)
                    if form is None or any(
                        source is not result.program_graph.one
                        and source.name not in ("in_read_addr_lo", "in_read_addr_hi")
                        for source in form
                    ):
                        raise RuntimeError(f"{label}: continuation repeats a prefix memory read")
        return result

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
    spec = lower_execution(execution, info)
    expert = {
        "kind": "exec_opcode" if expert_kind == "exec_resume" else expert_kind,
        "read_resume": read_resume,
        "name": f"shape_{name}",
        "shape_id": sid,
        "opcode_checks": checks,
        "length": len(template),
        "n_steps": n_steps,
        "model": model,
        "step_spec": spec,
        "output_slots": readout_slots(spec),
        **shape_timing(execution, self_target),
    }
    metadata = {
        "id": sid,
        "name": name,
        "sig": [[o, b] for o, b in checks],
        "length": len(template),
        "close": (
            "jmp" if self_target else "branch" if ops[-1].operation in BRANCH_OPERATIONS else "none"
        ),
    }
    return expert, metadata


def _build_symbolic_execution_experts(
    pg,
    shapes,
    *,
    expert_kind: Literal["shape", "exec_opcode", "exec_resume"],
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
    0..): per opcode, the interrupts, canonical shapes and bounded counter
    transitions; and the shapes' metadata."""
    experts = build_opcode_execution_experts(pg, sorted(OPCODES), resume_reads=True)
    experts.extend(build_interrupt_experts(pg))
    shape_experts, shapes_meta = build_shape_experts(pg, SHAPE_LIBRARY)
    return (
        experts
        + shape_experts
        + build_counter_recurrence_experts()
        + build_read_request_experts(pg)
    ), shapes_meta


def build_read_request_experts(pg) -> list[dict]:
    """Project address and access timing from the instruction graph.

    These weights run before an effectful read. They neither consume the
    operand byte nor commit architectural state, writes or interrupts.
    """
    dims = {dim.name: dim for dim in pg.all_dims}
    fields = ("ea", "cycles", "fetched_operand_lo", "fetched_operand_hi")
    request = pg.with_outputs({"READ_REQUEST": sum(dims[name] for name in fields)})
    folder = GraphFolder(request, root_names=fields, structural_reads=True)
    circuits = {}
    for byte, op in sorted(OPCODES.items()):
        if op.operation not in READS_MEMORY or op.mode not in MEMORY_MODES:
            continue
        pins = opcode_pins(byte, OPCODES) | {"current_halted": 0}
        folded = folder.fold(pins, label=f"read_{byte:02X}", inline_linear=True)
        if "operand_mem" in folded.dim_of_name:
            raise RuntimeError(f"read {byte:02X}: timing depends on the unserved operand")
        model, _, info = schedule_and_build(folded)

        def slot(name, info=info, folded=folded):
            return info.slot_of[folded.dim_of_name[name]]

        spec = step_spec_base(slot)
        spec["outputs"] = {
            "address": slot("ea"),
            "cycles": slot("cycles"),
            "oplo": slot("fetched_operand_lo"),
            "ophi": slot("fetched_operand_hi"),
        }
        # Share only byte-for-byte identical weights, read layouts and
        # embedding/readout contracts; opcode names are not a proof.
        signature = (
            repr(spec),
            model.one_slot,
            model.tok.weight.shape[1],
            tuple(
                tuple(
                    (name, array.shape, array.tobytes())
                    for name, array in sorted(layer_arrays(model, i).items())
                )
                for i in range(len(model.attn))
            ),
        )
        if signature not in circuits:
            circuits[signature] = {
                "kind": "read_request",
                "name": f"read_{byte:02X}",
                "bytes": [],
                "model": model,
                "step_spec": spec,
                "output_slots": readout_slots(spec),
            }
        circuits[signature]["bytes"].append(byte)
    return list(circuits.values())


def build_counter_recurrence_experts() -> list[dict]:
    """Compile bounded loop transitions through the same weight pipeline."""
    experts = []
    for operation, (reg, delta) in COUNTERS.items():
        register = "x" if reg == 1 else "y"
        pg = counter_recurrence_graph(register, delta)
        dims = {dim.name: dim for dim in pg.all_dims}
        changed = {
            f"new_{register}": dims[f"new_{register}"],
            "p_z": dims["loop_z"],
            "p_n": dims["loop_n"],
        }
        execution = ExecutionGraph(
            program_graph=pg,
            n_steps=0,
            memory_writes=(),
            outputs=tuple(
                ArchitecturalOutput(name, changed.get(name)) for name, _, _ in STATE_FIELDS
            ),
            halt_dimension=None,
            cycles=(dims["loop_elapsed"],),
            cycle_bounds=(255 * 255,),
            dim_of_name=dims,
        )
        model, _rows, info = schedule_and_build(execution)
        spec = lower_execution(execution, info)
        spec["outputs"] = {"iterations": info.slot_of[dims["loop_iterations"]]}
        experts.append(
            {
                "kind": "exec_loop",
                "name": f"loop_{operation}",
                "n_steps": 0,
                "reg": reg,
                "delta": delta,
                "model": model,
                "step_spec": spec,
                "output_slots": readout_slots(spec),
            }
        )
    return experts
