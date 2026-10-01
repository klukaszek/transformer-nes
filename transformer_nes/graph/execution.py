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
    target_dimension: Dimension | None = None
    relative_to_head: bool = False
    relative_operand_offset: int | None = None
    absolute_operand_offset: int | None = None
    flag_mask: int = 0
    flag_value: bool = True


@dataclass(frozen=True)
class ExecutionGraph:
    program_graph: ProgramGraph
    n_steps: int
    state_fields: tuple[StateField, ...]
    memory_reads: tuple[MemoryRead, ...]
    memory_writes: tuple[MemoryWrite, ...]
    outputs: tuple[ArchitecturalOutput, ...]
    control: ControlFlow
    halt_dimension: Dimension | None
    # Each step's cycle count: folded to a constant, or a model output;
    # and an upper bound on each.
    cycles: tuple[int | Dimension, ...]
    cycle_bounds: tuple[int, ...]
    dim_of_name: dict[str, Dimension]
    # Branch predictions to check before committing (a superblock's):
    # ("state", flag mask, expected) or ("model", dimension, expected).
    guards: tuple[tuple, ...] = ()

    def output(self, name: str) -> ArchitecturalOutput:
        return next(output for output in self.outputs if output.name == name)

    @property
    def cycles_prefix(self) -> int:
        """At most the cycles before the last step, which the run loop's
        budget guard needs ahead of dispatch: a window that fits even at
        this bound contains only instructions single steps would run."""
        return sum(self.cycle_bounds[:-1])
