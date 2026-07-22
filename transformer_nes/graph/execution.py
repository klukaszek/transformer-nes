"""Typed boundary between licensed proof folds and executable models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformer_nes.graph.core import (
    Dimension,
    InputDimension,
    MemoryRead,
    ProgramGraph,
)


@dataclass(frozen=True)
class StateField:
    name: str
    dimension: InputDimension


@dataclass(frozen=True)
class MemoryWrite:
    step: int
    order: int
    address: Dimension
    value: Dimension


@dataclass(frozen=True)
class ArchitecturalOutput:
    name: str
    source: Literal["model", "preserve", "constant"]
    dimension: Dimension | None = None
    constant: int | None = None


@dataclass(frozen=True)
class ControlFlow:
    predicate: Dimension | None
    fallthrough: int
    target: int
    relative_to_head: bool = False
    relative_operand_offset: int | None = None
    flag_mask: int = 0
    flag_value: bool = True


@dataclass(frozen=True)
class RecurrenceSummary:
    kind: Literal["identity", "affine_until", "invariant_poll"]
    steps_per_iteration: int
    cycles_per_iteration: int
    cycles_prefix: int
    register: Literal["x", "y"] | None = None
    delta: int = 0
    fallthrough_offset: int = 0
    closure_operand_offset: int | None = None
    closure_operand_value: int | None = None


@dataclass(frozen=True)
class ExecutionGraph:
    program_graph: ProgramGraph
    n_steps: int
    state_fields: tuple[StateField, ...]
    memory_reads: tuple[MemoryRead, ...]
    memory_writes: tuple[MemoryWrite, ...]
    outputs: tuple[ArchitecturalOutput, ...]
    control: ControlFlow
    recurrence: RecurrenceSummary | None
    dim_of_name: dict[str, Dimension]

    def output(self, name: str) -> ArchitecturalOutput:
        return next(output for output in self.outputs if output.name == name)
