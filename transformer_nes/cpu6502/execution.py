"""CPU execution effects and their lowering to the residual-slot ABI.

Proof folds retain the facts needed to validate a dispatch. Projection keeps
only architectural outputs, ordered writes, control and cycles; lowering then
assigns residual slots. Dictionaries are the final runtime/checkpoint format.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Literal

from transformer_nes.cpu6502.folds import const_root, interval_of
from transformer_nes.cpu6502.spec import (
    FLAG_B,
    FLAG_C,
    FLAG_D,
    FLAG_I,
    FLAG_N,
    FLAG_U,
    FLAG_V,
    FLAG_Z,
)
from transformer_nes.graph.core import Dimension, Expression, ProgramGraph, project_program_graph
from transformer_nes.graph.fold import linear_input_form


@dataclass(frozen=True)
class MemoryWrite:
    step: int
    order: int
    address: Dimension
    value: Dimension
    original: Dimension | None = None


@dataclass(frozen=True)
class ArchitecturalOutput:
    name: str
    dimension: Dimension | None = None
    constant: int | None = None

    def __post_init__(self):
        if self.dimension is not None and self.constant is not None:
            raise ValueError(f"{self.name}: an output cannot be both a model value and a constant")

    @property
    def source(self) -> Literal["model", "preserve", "constant"]:
        if self.dimension is not None:
            return "model"
        return "constant" if self.constant is not None else "preserve"


@dataclass(frozen=True)
class ExecutionGraph:
    program_graph: ProgramGraph
    n_steps: int
    memory_writes: tuple[MemoryWrite, ...]
    outputs: tuple[ArchitecturalOutput, ...]
    halt_dimension: Dimension | None
    # Each step's cycle count: folded to a constant, or a model output;
    # and an upper bound on each.
    cycles: tuple[int | Dimension, ...]
    cycle_bounds: tuple[int, ...]
    dim_of_name: dict[str, Dimension]

    @property
    def cycles_prefix(self) -> int:
        """At most the cycles before the last step, which the run loop's
        budget guard needs ahead of dispatch: a window that fits even at
        this bound contains only instructions single steps would run."""
        return sum(self.cycle_bounds[:-1])


# Residual input slots shared by every generated step spec: spec key -> the
# graph dimension name it resolves through (union slot map or fold name map).
INPUT_SLOT_NAMES = {
    "is_state": "is_state",
    "pc_lo": "in_pc_lo",
    "pc_hi": "in_pc_hi",
    "a": "in_a",
    "x": "in_x",
    "y": "in_y",
    "s": "in_s",
    "ir": "in_ir",
    "oplo": "in_oplo",
    "ophi": "in_ophi",
    "ea_lo": "in_ea_lo",
    "ea_hi": "in_ea_hi",
    "data": "in_data",
    "halted": "in_halted",
    "nmi": "in_nmi",
    "irq": "in_irq",
    "read_addr_lo": "in_read_addr_lo",
    "read_addr_hi": "in_read_addr_hi",
    "read_cycles": "in_read_cycles",
    "loop_limit": "in_loop_limit",
    "loop_cycles": "in_loop_cycles",
}
P_FLAG_NAMES = (
    ("in_p_c", FLAG_C),
    ("in_p_z", FLAG_Z),
    ("in_p_i", FLAG_I),
    ("in_p_d", FLAG_D),
    ("in_p_b", FLAG_B),
    ("in_p_u", FLAG_U),
    ("in_p_v", FLAG_V),
    ("in_p_n", FLAG_N),
)
# wide_bits builds only (absent from other graphs, so added when present):
# served A-bit state inputs, MSB first.
OPTIONAL_INPUT_SLOT_NAMES = {f"a_bit{b}": f"in_a_bit{b}" for b in range(7, -1, -1)}


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


SEQUENCE_FLAG_DIMS = (
    ("p_c", "new_p_c_final", FLAG_C, "in_p_c"),
    ("p_z", "new_p_z_final", FLAG_Z, "in_p_z"),
    ("p_i", "new_p_i_final", FLAG_I, "in_p_i"),
    ("p_d", "new_p_d_final", FLAG_D, "in_p_d"),
    ("p_b", "new_p_b_final", FLAG_B, "in_p_b"),
    ("p_v", "new_p_v_final", FLAG_V, "in_p_v"),
    ("p_n", "new_p_n_final", FLAG_N, "in_p_n"),
)

STATE_FIELDS = (
    ("new_pc", "next_pc", None),
    ("new_a", "new_a", "in_a"),
    ("new_x", "new_x", "in_x"),
    ("new_y", "new_y", "in_y"),
    ("new_s", "new_s", "in_s"),
    *((name, final, input_name) for name, final, _mask, input_name in SEQUENCE_FLAG_DIMS),
)

WRITE_RECEIPTS = (
    ("is_store", "write_addr_out", "write_value_out"),
    ("is_store2", "write_addr2_out", "write_value2_out"),
    ("is_store3", "write_addr3_out", "write_value3_out"),
)
SEQUENCE_ROOT_DIMS = (
    "new_a",
    "new_x",
    "new_y",
    "new_s",
    *(dim_name for _name, dim_name, _mask, _input_name in SEQUENCE_FLAG_DIMS),
    "ea",
    "next_pc",
    "new_halted",
    "branch_taken",
    "cycles",
    "rmw_write",
    "write_original_out",
    *(name for receipt in WRITE_RECEIPTS for name in receipt),
)


def architectural_outputs(fold_result, n_steps: int) -> tuple[ArchitecturalOutput, ...]:
    """Classify final state directly as model values, constants or preservation."""
    by_name, one = fold_result.dim_of_name, fold_result.program_graph.one
    outputs = []
    for name, final, input_name in STATE_FIELDS:
        dimension = by_name[f"{final}__s{n_steps - 1}"]
        form = linear_input_form(dimension.as_expr())
        if input_name is not None and form == {by_name[input_name]: 1.0}:
            output = ArchitecturalOutput(name)
        elif all(source is one for source in form):
            output = ArchitecturalOutput(name, constant=int(round(form.get(one, 0.0))))
        else:
            output = ArchitecturalOutput(name, dimension=dimension)
        outputs.append(output)
    return tuple(outputs)


def project_sequence_execution(
    fold_result,
    n_steps: int,
    halt_dimension: Dimension | None = None,
) -> ExecutionGraph:
    """Replace the proof graph's per-step trace ABI with its execution ABI.

    The complete fold is retained until all control, read, write and halt
    licenses have passed.  Only then do we keep the final architectural
    state and the address/value of each statically-live ordered write.  The
    projected graph is what gets scheduled and converted to weights.
    """
    outputs = architectural_outputs(fold_result, n_steps)
    output_tokens = {
        f"state:{output.name}": Expression({output.dimension: 1})
        for output in outputs
        if output.dimension is not None
    }

    writes = []
    for k in range(n_steps):
        for order, (flag, addr, value) in enumerate(WRITE_RECEIPTS):
            active = const_root(fold_result, flag, k, fold_result.label)
            if active not in (0, 1):
                raise RuntimeError(f"{fold_result.label} s{k}: {flag} is not boolean")
            if not active:
                continue
            address = fold_result.dim_of_name[f"{addr}__s{k}"]
            write_value = fold_result.dim_of_name[f"{value}__s{k}"]
            rmw = const_root(fold_result, "rmw_write", k, fold_result.label)
            if rmw not in (0, 1):
                raise RuntimeError(f"{fold_result.label} s{k}: RMW kind is not boolean")
            original = fold_result.dim_of_name[f"write_original_out__s{k}"] if order == 0 and rmw else None
            writes.append(MemoryWrite(k, order, address, write_value, original))
            output_tokens[f"write:{k}:{order}:address"] = Expression({address: 1})
            output_tokens[f"write:{k}:{order}:value"] = Expression({write_value: 1})
            if original is not None:
                output_tokens[f"write:{k}:{order}:original"] = Expression({original: 1})

    # Each step's cycles: folded to a constant where the pins decide them
    # (most opcodes), else a model output (taken branches, page crosses).
    # The bound: the folded value's interval, or for cycles that depend on
    # memory (an indirect read's page cross) the most an instruction
    # costs, 7 + the penalties.
    one = fold_result.program_graph.one
    cycles, bounds = [], []
    for k in range(n_steps):
        dim = fold_result.dim_of_name[f"cycles__s{k}"]
        form = linear_input_form(Expression({dim: 1.0}))
        if all(source is one for source in form):
            cycles.append(int(round(form.get(one, 0.0))))
            bounds.append(cycles[-1])
            continue
        cycles.append(dim)
        output_tokens[f"cycles:{k}"] = Expression({dim: 1})
        try:
            bounds.append(int(math.floor(interval_of(dim)[1])))
        except RuntimeError:
            bounds.append(9)

    if halt_dimension is not None:
        output_tokens["halt"] = Expression({halt_dimension: 1})
    program_graph = project_program_graph(
        fold_result.program_graph,
        output_tokens,
        retain_input_tokens=False,
    )
    dim_of_name = {}
    for dim in program_graph.all_dims:
        dim_of_name.setdefault(dim.name, dim)
    return ExecutionGraph(
        program_graph=program_graph,
        n_steps=n_steps,
        memory_writes=tuple(writes),
        outputs=tuple(outputs),
        halt_dimension=halt_dimension,
        cycles=tuple(cycles),
        cycle_bounds=tuple(bounds),
        dim_of_name=dim_of_name,
    )


def lower_execution(execution: ExecutionGraph, info) -> dict:
    """Lower graph dimensions to the slot-based runtime/checkpoint wire format."""

    def slot(name):
        return info.slot_of[execution.dim_of_name[name]]

    spec = step_spec_base(slot)
    spec["sequence_final"] = {
        "model": {
            output.name: info.slot_of[output.dimension]
            for output in execution.outputs
            if output.source == "model"
        },
        "preserve": [output.name for output in execution.outputs if output.source == "preserve"],
        "constant": {
            output.name: output.constant
            for output in execution.outputs
            if output.source == "constant"
        },
    }
    writes = [[] for _ in range(execution.n_steps)]
    for write in execution.memory_writes:
        writes[write.step].append(
            {"address": info.slot_of[write.address], "value": info.slot_of[write.value],
             "original": None if write.original is None else info.slot_of[write.original]}
        )
    spec["sequence_writes"] = writes
    spec["sequence_cycles"] = {
        "constant": sum(c for c in execution.cycles if isinstance(c, int)),
        "model": [info.slot_of[c] for c in execution.cycles if not isinstance(c, int)],
    }
    spec["sequence_halt"] = (
        None if execution.halt_dimension is None else info.slot_of[execution.halt_dimension]
    )
    return spec


def readout_slots(spec: dict) -> tuple[int, ...]:
    """Physical kernel outputs, in the order the native caller decodes them."""
    slots = set(spec.get("outputs", {}).values())
    if "sequence_final" in spec:
        slots.update(spec["sequence_final"]["model"].values())
        slots.update(s for writes in spec["sequence_writes"] for write in writes
                     for s in write.values() if s is not None)
        slots.update(spec["sequence_cycles"]["model"])
        if spec["sequence_halt"] is not None:
            slots.add(spec["sequence_halt"])
    return tuple(sorted(slots))


def shape_timing(execution: ExecutionGraph, spin: bool) -> dict:
    """How the run loop times a window.  Its budget guard needs a bound on
    the cycles before the last step up front (`cycles_prefix`); every
    step's exact cycles may stay a model output (a page cross, a closing
    branch).  A `JMP *` spin changes nothing per iteration, so the loop
    retires as many as the budget allows at `spin` cycles each (0 for any
    other window).  A loop window's recurrence is the run loop's
    (shapes.py)."""
    if spin and not isinstance(execution.cycles[0], int):
        raise RuntimeError("a JMP spin's cycles did not fold")
    return {
        "cycles_prefix": execution.cycles_prefix,
        "spin": execution.cycles[0] if spin else 0,
    }
