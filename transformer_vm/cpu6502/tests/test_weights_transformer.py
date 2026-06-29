"""The flagship claim: the 6502 executes as real transformer weights.

Builds the analytic weights (ASAP schedule, no training) and runs genuine
transformer inference -- per-position embeddings, Q/K/V projections, hull
hardmax attention, ReGLU FFN, output head -- lockstep against the
reference core.  Expensive (~2-4 minutes): dense float64 matvecs through
66 layers at d_model~1800 for every token position.
"""

from __future__ import annotations

import pytest

from transformer_vm.cpu6502 import reference
from transformer_vm.cpu6502.weights_runner import (
    VERIFY_START,
    Cpu6502TransformerMachine,
    verify_program,
)


@pytest.mark.slow
def test_transformer_weights_match_reference():
    machine = Cpu6502TransformerMachine(bcd=False, cache="hull")
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=15)
    assert len(trace.rows) == 15

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = VERIFY_START
    for i, row in enumerate(trace):
        ref_row = ref.step()
        a, b = ref_row.state, row.state
        assert (a.pc, a.a, a.x, a.y, a.s, a.p) == (b.pc, b.a, b.x, b.y, b.s, b.p), (
            f"step {i}: ref={a} transformer={b}"
        )
        assert {(w.addr, w.value) for w in ref_row.writes} == {
            (w.addr, w.value) for w in row.writes
        }, f"step {i} writes"
