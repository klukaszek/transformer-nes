"""Shared tables for native CPU run loops."""

from __future__ import annotations

from transformer_vm.cpu6502.spec import (
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
    READS_MEMORY,
)

_READ_MODE_CODE = {
    MODE_ZERO_PAGE: 1,
    MODE_ZERO_PAGE_X: 2,
    MODE_ZERO_PAGE_Y: 3,
    MODE_ABSOLUTE: 4,
    MODE_ABSOLUTE_X: 5,
    MODE_ABSOLUTE_Y: 6,
    MODE_INDIRECT_X: 7,
    MODE_INDIRECT_Y: 8,
}


def read_mode_table() -> tuple[int, ...]:
    """Opcode-to-addressing-mode codes used by MMIO shadow decoding."""
    table = [0] * 256
    for byte, op in OPCODES.items():
        if op.operation in READS_MEMORY:
            table[byte] = _READ_MODE_CODE.get(op.mode, 0)
    return tuple(table)


def cycle_table() -> tuple[int, ...]:
    """Base opcode cycles; dynamic penalties are outside the run protocol."""
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = op.cycles
    return tuple(table)
