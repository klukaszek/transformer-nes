"""Canonical trace format shared by the reference and weights runners."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from transformer_nes.cpu6502.spec import (
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_ACCUMULATOR,
    MODE_IMMEDIATE,
    MODE_INDIRECT,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_RELATIVE,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
)


@dataclass(frozen=True)
class Cpu6502State:
    """A snapshot of the 6502 CPU state.

    The ``ir``, ``operand_lo``, ``operand_hi``, ``ea``, and ``data`` fields
    are part of the internal instruction-fetch / execute machinery so the
    reference trace and generated engine share one state contract.
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
    original: int = -1  # RMW operand; -1 for an ordinary/stack write


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


# ── The tables a run loop traces instructions by ─────────────────
# (native/console/cpu.cpp trace_ea / trace_data): addressing mode,
# instruction length and the data a trace row reports, by opcode.

ADDRESS_MODE_CODE = {
    MODE_ZERO_PAGE: 1,
    MODE_ZERO_PAGE_X: 2,
    MODE_ZERO_PAGE_Y: 3,
    MODE_ABSOLUTE: 4,
    MODE_ABSOLUTE_X: 5,
    MODE_ABSOLUTE_Y: 6,
    MODE_INDIRECT_X: 7,
    MODE_INDIRECT_Y: 8,
}


TRACE_MODE_CODE = ADDRESS_MODE_CODE | {
    MODE_IMMEDIATE: 11,
    MODE_INDIRECT: 9,
    MODE_RELATIVE: 10,
}


def trace_mode_table() -> tuple[int, ...]:
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = TRACE_MODE_CODE.get(op.mode, 0)
    return tuple(table)


def trace_length_table() -> tuple[int, ...]:
    table = [1] * 256
    for byte, op in OPCODES.items():
        table[byte] = op.length
    return tuple(table)


def trace_data_table() -> tuple[int, ...]:
    """Trace source codes consumed by the runtime template's trace_data()."""
    kinds = {
        1: ("store_a", "pha"),
        2: ("store_x",),
        3: ("store_y",),
        4: ("php",),
        5: ("inc", "isb", "dec", "dcp", "sax", "slo", "rla", "sre", "rra"),
        6: ("load_a", "lax", "adc", "sbc", "and", "ora", "eor", "pla", "txa", "tya"),
        7: ("load_x", "inx", "dex", "tax", "tsx"),
        8: ("load_y", "iny", "dey", "tay"),
        9: ("txs",),
        10: ("plp",),
        11: ("cmp",),
        12: ("cpx",),
        13: ("cpy",),
        14: ("bit",),
    }
    codes = {operation: kind for kind, operations in kinds.items() for operation in operations}
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = (
            (6 if op.mode == MODE_ACCUMULATOR else 5)
            if op.operation in {"asl", "lsr", "rol", "ror"}
            else codes.get(op.operation, 0)
        )
    return tuple(table)
