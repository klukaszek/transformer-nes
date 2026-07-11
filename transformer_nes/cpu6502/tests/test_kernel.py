"""Verify the generated C++ kernel against the Python graph evaluator."""

from __future__ import annotations

import pytest

from transformer_nes.cpu6502.evaluator import Cpu6502Runtime

PROGRAMS = {
    "jsr_rts": bytes([0x20, 0x06, 0x80, 0x00, 0xEA, 0xEA, 0xA9, 0x42, 0x60]),
    "indirect": bytes([
        0xA9, 0x00, 0x85, 0x10, 0xA9, 0x03, 0x85, 0x11, 0xA9, 0x77,
        0x8D, 0x00, 0x03, 0xA2, 0x02, 0xA1, 0x0E, 0x00,
    ]),
    "decimal": bytes([0xF8, 0x18, 0xA9, 0x04, 0x69, 0x05, 0x00]),
}


@pytest.mark.slow
def test_cpp_kernel_matches_python_graph():
    from transformer_nes.cpu6502.kernel import Cpu6502CppRuntime

    cpp = Cpu6502CppRuntime(bcd=True)
    for name, program in PROGRAMS.items():
        image = bytearray(0x10000)
        image[0x8000:0x8000 + len(program)] = program
        py_trace = Cpu6502Runtime().run_image(bytes(image), start_pc=0x8000, max_steps=20)
        cpp_trace = cpp.run_image(bytes(image), start_pc=0x8000, max_steps=20)
        assert len(py_trace.rows) == len(cpp_trace.rows), name
        for a, b in zip(py_trace.rows, cpp_trace.rows, strict=True):
            assert a.state == b.state, f"{name}: {a}\nvs {b}"
            assert a.writes == b.writes, name
