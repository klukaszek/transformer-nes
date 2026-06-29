"""Canonical trace format shared by reference, graph, and transformer runners."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cpu6502State:
    """A snapshot of the 6502 CPU state.

    The ``ir``, ``operand_lo``, ``operand_hi``, ``ea``, and ``data`` fields
    are part of the internal instruction-fetch / execute machinery so that
    the graph interpreter and reference can emit identical intermediate
    state.
    """

    pc: int
    a: int
    x: int
    y: int
    s: int
    p: int
    ir: int = 0  # instruction register (opcode byte)
    operand_lo: int = 0
    operand_hi: int = 0
    ea: int = 0  # effective address
    data: int = 0  # data bus / ALU result
    halted: bool = False


@dataclass(frozen=True)
class MemWrite:
    """A single byte write to memory, emitted as part of a trace row."""

    step: int
    addr: int
    value: int


@dataclass(frozen=True)
class TraceRow:
    """One row in an instruction-level execution trace."""

    step: int
    state: Cpu6502State
    writes: tuple[MemWrite, ...] = ()

    @property
    def pc(self) -> int:
        return self.state.pc

    @property
    def opcode(self) -> int:
        return self.state.ir


def format_trace_row(row: TraceRow) -> str:
    """Format a trace row like the project canonical trace contract.

    Example:
        step=000 PC=8000 OP=A9 A=00 X=00 Y=00 S=FD P=24
        step=002 PC=8004 OP=85 A=08 X=00 Y=00 S=FD P=24 write[$00]=08
    """
    s = row.state
    parts = [
        f"step={row.step:03d}",
        f"PC={s.pc:04X}",
        f"OP={s.ir:02X}",
        f"A={s.a:02X}",
        f"X={s.x:02X}",
        f"Y={s.y:02X}",
        f"S={s.s:02X}",
        f"P={s.p:02X}",
    ]
    for w in row.writes:
        parts.append(f"write[${w.addr:04X}]={w.value:02X}")
    return " ".join(parts)


def format_trace(trace: Iterable[TraceRow]) -> str:
    """Return a multi-line string for a trace."""
    return "\n".join(format_trace_row(row) for row in trace)


@dataclass
class Trace:
    """Collected trace of a program run."""

    rows: list[TraceRow] = field(default_factory=list)

    def append(self, row: TraceRow) -> None:
        self.rows.append(row)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TraceRow:
        return self.rows[index]

    def final_state(self) -> Cpu6502State | None:
        return self.rows[-1].state if self.rows else None

    def writes(self) -> list[MemWrite]:
        return [w for row in self.rows for w in row.writes]

    def format(self) -> str:
        return format_trace(self.rows)
