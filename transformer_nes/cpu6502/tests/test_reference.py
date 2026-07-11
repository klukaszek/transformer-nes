"""Tests for the boring Python 6502 reference core."""

from __future__ import annotations

import os

import pytest

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.spec import FLAG_C, FLAG_N, FLAG_Z, asm
from transformer_nes.cpu6502.trace import format_trace_row

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "tiny")


def example(name: str) -> bytes:
    with open(os.path.join(EXAMPLES_DIR, name), "rb") as f:
        return f.read()


# ── Program 1: load immediate ─────────────────────────────────────


def test_lda_imm_sets_a_and_flags():
    cpu = reference.Cpu6502()
    trace = cpu.run(example("load_immediate.bin"), start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final is not None
    assert final.a == 0x42
    assert final.p & FLAG_N == 0
    assert final.p & FLAG_Z == 0
    assert final.halted


# ── Program 2 & 3: zero / negative flags ──────────────────────────


def test_lda_zero_sets_z_flag():
    trace = reference.run(example("zero_flag.bin"), start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.a == 0x00
    assert final.p & FLAG_Z
    assert (final.p & FLAG_N) == 0


def test_lda_negative_sets_n_flag():
    trace = reference.run(example("negative_flag.bin"), start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.a == 0x80
    assert final.p & FLAG_N
    assert (final.p & FLAG_Z) == 0


# ── Program 4: store zero page ────────────────────────────────────


def test_sta_zpg_writes_memory():
    cpu = reference.Cpu6502()
    trace = cpu.run(example("store_zpg.bin"), start_pc=0x8000, max_steps=10)
    assert cpu.memory[0x0010] == 0x37
    # The trace also contains the BRK stack pushes; check the STA write.
    sta_writes = [w for w in trace.writes() if w.addr == 0x0010]
    assert len(sta_writes) == 1
    assert sta_writes[0].value == 0x37


# ── Program 5: branch loop ────────────────────────────────────────


def test_bne_loop_counts_down():
    trace = reference.run(example("branch_loop.bin"), start_pc=0x8000, max_steps=20)
    final = trace.final_state()
    assert final.x == 0x00
    assert final.p & FLAG_Z
    assert final.halted
    # Three DEX iterations plus fetch of final BRK => expect branch taken 3x.
    taken_branches = [r for r in trace.rows if r.opcode == 0xD0 and r.state.pc != r.pc + 2]
    assert len(taken_branches) == 3


# ── Acceptance-criteria opcode subset ─────────────────────────────


def test_ldx_imm():
    program = asm().ldx_imm(0xAB).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.x == 0xAB
    assert final.p & FLAG_N


def test_ldy_imm():
    program = asm().ldy_imm(0x05).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.y == 0x05
    assert (final.p & FLAG_Z) == 0


def test_sta_abs():
    program = asm().lda_imm(0x55).sta_abs(0x0200).brk().bytes()
    cpu = reference.Cpu6502()
    trace = cpu.run(program, start_pc=0x8000, max_steps=10)
    assert cpu.memory[0x0200] == 0x55
    writes = trace.writes()
    assert any(w.addr == 0x0200 and w.value == 0x55 for w in writes)


def test_adc_imm():
    # A=5 + 7 = 0x0C, no carry in
    program = asm().lda_imm(0x05).adc_imm(0x07).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.a == 0x0C
    assert (final.p & FLAG_C) == 0
    assert (final.p & FLAG_Z) == 0
    assert (final.p & FLAG_N) == 0


def test_adc_imm_with_carry():
    # A=0xFF + 1 + carry=1 => 0x01 with C=1, V=0
    program = asm().lda_imm(0xFF).adc_imm(0x01).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.a == 0x00
    assert final.p & FLAG_C
    assert final.p & FLAG_Z
    assert (final.p & FLAG_N) == 0


def test_sbc_imm():
    # A=10 - 3.  With C=0 initially the operation is A - M - 1 = 6.
    # No unsigned borrow is required, so carry is set afterward.
    program = asm().lda_imm(0x0A).sbc_imm(0x03).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.a == 0x06
    assert final.p & FLAG_C  # no borrow


def test_inx():
    program = asm().ldx_imm(0x7F).inx().brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.x == 0x80
    assert final.p & FLAG_N
    assert (final.p & FLAG_Z) == 0


def test_dex_wrap():
    program = asm().ldx_imm(0x00).dex().brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.x == 0xFF
    assert final.p & FLAG_N


def test_tax():
    program = asm().lda_imm(0x00).tax().brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    assert final.x == 0x00
    assert final.p & FLAG_Z


def test_jmp_abs():
    program = asm().jmp_abs(0x8005).lda_imm(0xAA).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    # Should skip over LDA #AA and halt at BRK
    assert final.a == 0x00
    assert final.halted


def test_beq_taken():
    program = asm().lda_imm(0x00).beq(0x02).lda_imm(0xFF).brk().bytes()
    trace = reference.run(program, start_pc=0x8000, max_steps=10)
    final = trace.final_state()
    # BEQ taken skips LDA #FF, so A stays 0
    assert final.a == 0x00
    assert final.halted


# ── Trace formatting ──────────────────────────────────────────────


def test_trace_format_matches_contract():
    # Trace rows are post-execution snapshots labeled with the address of the
    # instruction that was just executed (PC has not yet advanced in the row).
    program = asm().lda_imm(0x05).adc_imm(0x03).sta_zpg(0x00).brk().bytes()
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
