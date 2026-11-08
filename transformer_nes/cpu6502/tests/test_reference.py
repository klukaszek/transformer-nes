"""Tests for the boring Python 6502 reference core."""

from __future__ import annotations

import pytest

from transformer_nes.cpu6502 import reference
from transformer_nes.cpu6502.spec import (
    FLAG_C,
    FLAG_N,
    FLAG_Z,
    MODE_ABSOLUTE_X,
    opcode_for,
)
from transformer_nes.cpu6502.trace import format_trace_row

# ── Program 1: load immediate ─────────────────────────────────────


def test_lda_imm_sets_a_and_flags():
    cpu = reference.Cpu6502()
    trace = cpu.run(bytes.fromhex("A9 42 00"), start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x42
    assert final.p & FLAG_N == 0
    assert final.p & FLAG_Z == 0
    assert final.halted


# ── Program 2 & 3: zero / negative flags ──────────────────────────


def test_lda_zero_sets_z_flag():
    trace = reference.run(bytes.fromhex("A9 00 00"), start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x00
    assert final.p & FLAG_Z
    assert (final.p & FLAG_N) == 0


def test_lda_negative_sets_n_flag():
    trace = reference.run(bytes.fromhex("A9 80 00"), start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x80
    assert final.p & FLAG_N
    assert (final.p & FLAG_Z) == 0


# ── Program 4: store zero page ────────────────────────────────────


def test_sta_zpg_writes_memory():
    cpu = reference.Cpu6502()
    trace = cpu.run(bytes.fromhex("A9 37 85 10 00"), start_pc=0x8000, max_steps=10)
    assert cpu.memory[0x0010] == 0x37
    # The trace also contains the BRK stack pushes; check the STA write.
    sta_writes = [w for w in trace.writes() if w.addr == 0x0010]
    assert len(sta_writes) == 1
    assert sta_writes[0].value == 0x37


# ── Program 5: branch loop ────────────────────────────────────────


def test_bne_loop_counts_down():
    trace = reference.run(bytes.fromhex("A2 03 CA D0 FD 00"), start_pc=0x8000, max_steps=20)
    final = trace.final_state()
    assert final is not None
    assert final.x == 0x00
    assert final.p & FLAG_Z
    assert final.halted
    # Three DEX iterations plus fetch of final BRK => expect branch taken 3x.
    taken_branches = [r for r in trace.rows if r.opcode == 0xD0 and r.state.pc != r.pc + 2]
    assert len(taken_branches) == 3


def test_dynamic_cycles_include_branch_and_indexed_read_penalties():
    cpu = reference.Cpu6502()
    cpu.memory[0x80FE:0x8100] = bytes([0xD0, 0x80])
    branch = opcode_for(0xD0)
    assert reference.instruction_cycles(cpu, branch, 0x80FE, 0, 0, 0) == 4
    assert reference.instruction_cycles(cpu, branch, 0x80FE, 0, 0, FLAG_Z) == 2

    cpu.memory[0x8000:0x8003] = bytes([0xBD, 0xFF, 0x20])
    indexed = opcode_for(0xBD)
    assert indexed.mode == MODE_ABSOLUTE_X
    assert reference.instruction_cycles(cpu, indexed, 0x8000, 1, 0, 0) == 5


def test_nes_internal_ram_mirrors_are_write_through():
    cpu = reference.Cpu6502(mirror_cpu_ram=True)
    cpu.write_memory(0x1803, 0xA7)
    assert [cpu.read_memory(addr) for addr in (0x0003, 0x0803, 0x1003, 0x1803)] == [0xA7] * 4


# ── Acceptance-criteria opcode subset ─────────────────────────────


def test_ldx_imm():
    program = bytes.fromhex("A2 AB 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.x == 0xAB
    assert final.p & FLAG_N


def test_ldy_imm():
    program = bytes.fromhex("A0 05 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.y == 0x05
    assert (final.p & FLAG_Z) == 0


def test_sta_abs():
    program = bytes.fromhex("A9 55 8D 00 02 00")
    cpu = reference.Cpu6502()
    trace = cpu.run(program, start_pc=0x8000, max_steps=10)
    assert cpu.memory[0x0200] == 0x55
    writes = trace.writes()
    assert any(w.addr == 0x0200 and w.value == 0x55 for w in writes)


def test_adc_imm():
    # A=5 + 7 = 0x0C, no carry in
    program = bytes.fromhex("A9 05 69 07 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x0C
    assert (final.p & FLAG_C) == 0
    assert (final.p & FLAG_Z) == 0
    assert (final.p & FLAG_N) == 0


def test_adc_imm_with_carry():
    # A=0xFF + 1 with carry initially clear => 0x00 with C=1, V=0
    program = bytes.fromhex("A9 FF 69 01 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x00
    assert final.p & FLAG_C
    assert final.p & FLAG_Z
    assert (final.p & FLAG_N) == 0


def test_sbc_imm():
    # A=10 - 3.  With C=0 initially the operation is A - M - 1 = 6.
    # No unsigned borrow is required, so carry is set afterward.
    program = bytes.fromhex("A9 0A E9 03 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x06
    assert final.p & FLAG_C  # no borrow


def test_inx():
    program = bytes.fromhex("A2 7F E8 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.x == 0x80
    assert final.p & FLAG_N
    assert (final.p & FLAG_Z) == 0


def test_dex_wrap():
    program = bytes.fromhex("A2 00 CA 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.x == 0xFF
    assert final.p & FLAG_N


def test_tax():
    program = bytes.fromhex("A9 00 AA 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.x == 0x00
    assert final.p & FLAG_Z


def test_jmp_abs():
    program = bytes.fromhex("4C 05 80 A9 AA 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    # Should skip over LDA #AA and halt at BRK
    assert final.a == 0x00
    assert final.halted


def test_beq_taken():
    program = bytes.fromhex("A9 00 F0 02 A9 FF 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    # BEQ taken skips LDA #FF, so A stays 0
    assert final.a == 0x00
    assert final.halted


# ── Trace formatting ──────────────────────────────────────────────


def test_trace_format_matches_contract():
    # Trace rows are post-execution snapshots labeled with the address of the
    # instruction that was just executed (PC has not yet advanced in the row).
    program = bytes.fromhex("A9 05 69 03 85 00 00")
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    lines = trace.format().splitlines()
    assert len(lines) == 4
    assert lines[0] == "step=000 PC=8000 OP=A9 A=05 X=00 Y=00 S=FD P=24"
    assert lines[2] == "step=002 PC=8004 OP=85 A=08 X=00 Y=00 S=FD P=24 write[$0000]=08"


def test_format_row_with_write():
    from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, TraceRow

    state = Cpu6502State(pc=0x8004, a=0x08, x=0, y=0, s=0xFD, p=0x24, ir=0x85)
    row = TraceRow(2, state, (MemWrite(2, 0x0000, 0x08),))
    text = format_trace_row(row)
    assert text == "step=002 PC=8004 OP=85 A=08 X=00 Y=00 S=FD P=24 write[$0000]=08"


# ── Unsupported opcode error ──────────────────────────────────────


def test_unsupported_opcode_raises():
    program = bytes([0x02, 0x00])  # KIL (unofficial)
    cpu = reference.Cpu6502()
    with pytest.raises(ValueError, match="unsupported opcode 0x02"):
        cpu.run(program, start_pc=0x8000, max_steps=5)
