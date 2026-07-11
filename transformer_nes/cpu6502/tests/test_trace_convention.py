"""The trace convention is the reference's, field for field.

Since 2026-07-09 the graph reports trace context exactly as reference.py
does: operands gated by instruction length (raw on an NMI step), ea per
addressing mode, and data as the operation's primary result (loads report
the loaded value, stores/RMW the written byte, compares the difference,
pure control flow 0).  This is what lets a routed expert fold away memory
reads whose only consumer was the old raw-read trace convention.

Checked instruction-by-instruction through the C++ kernel (same graph
semantics as every backend) against the reference oracle: full nestest
(all official + unofficial opcodes, 2A03) and a Klaus prefix (decimal
mode, BRK/RTI, stack wrap).
"""

import pytest

from transformer_nes.cpu6502.backends.kernel import Cpu6502CppRuntime
from transformer_nes.cpu6502.backends.reference import Cpu6502


def _assert_full_trace_match(image, start_pc, max_steps, *, bcd):
    rt = Cpu6502CppRuntime(bcd=bcd)
    out = rt.run_image_arrays(bytes(image), start_pc=start_pc, max_steps=max_steps)
    ref = Cpu6502(memory=bytearray(image), bcd_enabled=bcd)
    ref.pc = start_pc
    n = len(out["pc"])
    assert n > 0
    for i in range(n):
        s = ref.step().state
        got = tuple(int(out[f][i]) for f in
                    ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi", "ea", "data"))
        want = (s.pc, s.a, s.x, s.y, s.s, s.p, s.ir,
                s.operand_lo, s.operand_hi, s.ea, s.data)
        assert got == want, f"step {i}: kernel={got} ref={want}"
    return n


@pytest.mark.slow
def test_nestest_trace_fields_match_reference():
    from transformer_nes.nes.nestest import START_PC, load_image

    n = _assert_full_trace_match(load_image(), START_PC, 8991, bcd=False)
    assert n == 8991


@pytest.mark.slow
def test_klaus_prefix_trace_fields_match_reference():
    from transformer_nes.cpu6502.klaus import START_PC, load_image

    _assert_full_trace_match(load_image(), START_PC, 200_000, bcd=True)
