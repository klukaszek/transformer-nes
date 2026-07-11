"""Compare graph-evaluated 6502 traces against the reference core."""

from __future__ import annotations

import os

import pytest

from transformer_nes.cpu6502.backends.evaluator import Cpu6502Runtime
from transformer_nes.cpu6502.backends.evaluator import run_program as graph_run
from transformer_nes.cpu6502.backends.reference import run as ref_run
from transformer_nes.cpu6502.spec import asm

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "tiny")


def example(name: str) -> bytes:
    with open(os.path.join(EXAMPLES_DIR, name), "rb") as f:
        return f.read()


def traces_equal(ref_trace, graph_trace):
    if len(ref_trace.rows) != len(graph_trace.rows):
        return False
    for r1, r2 in zip(ref_trace.rows, graph_trace.rows, strict=True):
        s1, s2 = r1.state, r2.state
        # Full-state comparison: since 2026-07-09 the graph reports trace
        # context (ir, operands, ea, data) with the reference's exact
        # conventions, so every field must match.
        if s1 != s2:
            return False
        if r1.writes != r2.writes:
            return False
    return True


@pytest.mark.parametrize(
    "name",
    [
        "load_immediate.bin",
        "zero_flag.bin",
        "negative_flag.bin",
        "store_zpg.bin",
        "branch_loop.bin",
    ],
)
def test_tiny_examples_match_reference(name: str):
    program = example(name)
    ref = ref_run(program, start_pc=0x8000, max_steps=100)
    graph = graph_run(
        os.path.join(EXAMPLES_DIR, name),
        start_pc=0x8000,
        max_steps=100,
    )
    assert traces_equal(ref, graph), (
        f"mismatch for {name}\nREF:\n{ref.format()}\nGRAPH:\n{graph.format()}"
    )


def test_ldx_imm():
    program = asm().ldx_imm(0xAB).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_ldy_imm():
    program = asm().ldy_imm(0x05).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_sta_abs():
    program = asm().lda_imm(0x55).sta_abs(0x0200).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_store_then_load_uses_updated_memory():
    program = asm().lda_imm(0x42).sta_zpg(0x10).lda_imm(0x00).raw(0xA5, 0x10).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_sparse_image_reads_missing_bytes_as_zero():
    image = bytearray(0x10000)
    image[0x0400:0x0404] = bytes([0xAD, 0x00, 0x30, 0x00])
    image[0x3001] = 0x77

    graph = Cpu6502Runtime().run_image(image, start_pc=0x0400, max_steps=2)

    assert graph.rows[0].state.a == 0
    assert graph.rows[0].state.data == 0


def test_adc_imm():
    program = asm().lda_imm(0x05).adc_imm(0x07).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_sbc_imm():
    program = asm().lda_imm(0x0A).sbc_imm(0x03).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_cpy_imm_updates_negative_and_zero_flags():
    program = asm().ldy_imm(0xFD).raw(0xC0, 0x00).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_inx():
    program = asm().ldx_imm(0x7F).inx().brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_dex_wrap():
    program = asm().ldx_imm(0x00).dex().brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_tax():
    program = asm().lda_imm(0x00).tax().brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_jmp_abs():
    program = asm().jmp_abs(0x8005).lda_imm(0xAA).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def test_beq_taken():
    program = asm().lda_imm(0x00).beq(0x02).lda_imm(0xFF).brk().bytes()
    ref = ref_run(program, start_pc=0x8000, max_steps=10)
    graph = run_graph(program)
    assert traces_equal(ref, graph)


def run_graph(program: bytes):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        return graph_run(path, start_pc=0x8000, max_steps=100)
    finally:
        os.unlink(path)


# ── Decimal mode ──────────────────────────────────────────────────
# Boundary nibble sums: al == 9 historically produced fractional stepglu
# outputs (thresholds must be integers), and V in decimal mode must come
# from the binary sum on both cores.

BCD_CASES = [
    (0x69, 0x18, 0x04, 0x05),  # ADC, C=0: low nibble sum exactly 9
    (0x69, 0x18, 0x09, 0x00),
    (0x69, 0x38, 0x90, 0x99),  # ADC, C=1: high-nibble carry + binary V
    (0x69, 0x18, 0x99, 0x01),
    (0xE9, 0x18, 0x28, 0x99),  # SBC borrow chains
    (0xE9, 0x38, 0x10, 0x05),
    (0xE9, 0x18, 0x00, 0x01),
]


def test_decimal_adc_sbc_match_reference():
    rt = Cpu6502Runtime()
    for op, carry, a, m in BCD_CASES:
        program = bytes([0xF8, carry, 0xA9, a, op, m, 0x00])
        ref = ref_run(program, start_pc=0x8000, max_steps=8)
        rt.reset()
        graph = rt.run(list(program), start_pc=0x8000, max_steps=8)
        assert traces_equal(ref, graph), (
            f"BCD mismatch op={op:02X} carry={carry:02X} A={a:02X} M={m:02X}\n"
            f"REF:\n{ref.format()}\nGRAPH:\n{graph.format()}"
        )
