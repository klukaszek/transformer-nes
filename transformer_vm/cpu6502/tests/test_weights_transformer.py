"""The flagship claim: the 6502 executes as real transformer weights.

Builds the analytic weights (ASAP schedule, no training) and runs genuine
transformer inference -- per-position embeddings, Q/K/V projections, hull
hardmax attention, ReGLU FFN, output head -- lockstep against the
reference core.  Expensive: real inference through the default float32 C++
engine, plus diagnostic comparisons against alternate matmul paths.
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
    machine = Cpu6502TransformerMachine(bcd=False)
    assert machine.matmul == "engine"
    assert machine.dtype == "float32"
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=30)
    assert len(trace.rows) == 30
    _lockstep_vs_reference(trace, image)


@pytest.mark.slow
def test_transformer_float32_engine_matches_high_address_nestest_prefix():
    from transformer_vm.cpu6502.nestest import START_PC, load_image

    image = load_image()
    machine = Cpu6502TransformerMachine(bcd=False)
    assert machine.matmul == "engine"
    assert machine.dtype == "float32"
    trace = machine.run_image(bytes(image), start_pc=START_PC, max_steps=20)

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = START_PC
    for i, row in enumerate(trace):
        ref_row = ref.step()
        a, b = ref_row.state, row.state
        assert (a.pc, a.a, a.x, a.y, a.s, a.p) == (b.pc, b.a, b.x, b.y, b.s, b.p), (
            f"step {i}: ref={a} transformer={b}"
        )
        assert ref_row.writes == row.writes, f"step {i} writes"


@pytest.mark.slow
def test_all_matmul_paths_produce_identical_traces():
    """CSR matvecs and the C++ engine are storage/dispatch changes, not
    semantics changes: dense torch, scipy CSR, and the engine must agree."""
    image = verify_program()
    traces = {}
    for matmul in ("dense", "sparse", "engine"):
        machine = Cpu6502TransformerMachine(
            bcd=False, cache="hull", matmul=matmul, dtype="float64"
        )
        traces[matmul] = machine.run_image(image, start_pc=VERIFY_START, max_steps=10)
    for other in ("sparse", "engine"):
        for a, b in zip(traces["dense"].rows, traces[other].rows, strict=True):
            assert a.state == b.state, other
            assert a.writes == b.writes, other
    _lockstep_vs_reference(traces["engine"], image)


@pytest.mark.slow
def test_transformer_true_softmax_matches_reference():
    machine = Cpu6502TransformerMachine(bcd=False, cache="standard", matmul="sparse")
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=20)
    _lockstep_vs_reference(trace, image)
