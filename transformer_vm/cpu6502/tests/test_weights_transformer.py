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


def _lockstep_vs_reference(trace, image):
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


@pytest.mark.slow
def test_transformer_weights_match_reference():
    machine = Cpu6502TransformerMachine(bcd=False, cache="hull", matmul="sparse")
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=30)
    assert len(trace.rows) == 30
    _lockstep_vs_reference(trace, image)


@pytest.mark.slow
def test_sparse_and_dense_paths_produce_identical_traces():
    """CSR matvecs are a storage change, not a semantics change."""
    image = verify_program()
    sparse = Cpu6502TransformerMachine(bcd=False, cache="hull", matmul="sparse")
    sparse_trace = sparse.run_image(image, start_pc=VERIFY_START, max_steps=10)
    dense = Cpu6502TransformerMachine(bcd=False, cache="hull", matmul="dense")
    dense_trace = dense.run_image(image, start_pc=VERIFY_START, max_steps=10)
    for a, b in zip(sparse_trace.rows, dense_trace.rows, strict=True):
        assert a.state == b.state
        assert a.writes == b.writes
    _lockstep_vs_reference(sparse_trace, image)


@pytest.mark.slow
def test_transformer_true_softmax_matches_reference():
    machine = Cpu6502TransformerMachine(bcd=False, cache="standard", matmul="sparse")
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=20)
    _lockstep_vs_reference(trace, image)
