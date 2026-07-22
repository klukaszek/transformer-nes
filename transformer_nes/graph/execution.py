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
class ExecutionGraph:
    program_graph: ProgramGraph
    n_steps: int
    state_fields: tuple[StateField, ...]
    memory_reads: tuple[MemoryRead, ...]
    memory_writes: tuple[MemoryWrite, ...]
    outputs: tuple[ArchitecturalOutput, ...]
    dim_of_name: dict[str, Dimension]

    def output(self, name: str) -> ArchitecturalOutput:
        return next(output for output in self.outputs if output.name == name)
