"""The flagship claim: the 6502 executes as real transformer weights.

Builds the analytic weights (ASAP schedule, no training) and runs genuine
transformer inference -- per-position embeddings, Q/K/V projections, hull
hardmax attention, ReGLU FFN, output head -- lockstep against the
reference core.  Expensive: real inference through the default float32
codegen C++ engine, cross-checked against the generic CSR engine.
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
    assert machine.matmul == "codegen"
    assert machine.dtype == "float32"
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=30)
    assert len(trace.rows) == 30
    _lockstep_vs_reference(trace, image)


@pytest.mark.slow
def test_transformer_float32_engine_matches_high_address_nestest_prefix():
    from transformer_vm.nes.nestest import START_PC, load_image

    image = load_image()
    machine = Cpu6502TransformerMachine(bcd=False)
    assert machine.matmul == "codegen"
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
def test_checkpoint_rebuild_is_invisible():
    """Snapshot-and-rebuild mid-run: the hull logs reseed from the RAM
    mirror without perturbing the executed trace.  Uses the nestest prefix
    so high-address reads cross checkpoint boundaries."""
    from transformer_vm.nes.nestest import START_PC, load_image

    image = load_image()
    machine = Cpu6502TransformerMachine(bcd=False, checkpoint_tokens=8)
    machine.boot_image(list(image), start_pc=START_PC)
    rows = [machine.machine_step() for _ in range(10)]
    assert machine.checkpoint()  # explicit rebuild between steps
    rows += [machine.machine_step() for _ in range(10)]
    assert machine.checkpoints >= 2  # the tiny auto threshold fired too

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = START_PC
    for i, row in enumerate(rows):
        assert row.step == i  # trace numbering survives rebuilds
        ref_row = ref.step()
        a, b = ref_row.state, row.state
        assert (a.pc, a.a, a.x, a.y, a.s, a.p) == (b.pc, b.a, b.x, b.y, b.s, b.p), (
            f"step {i}: ref={a} transformer={b}"
        )
        assert ref_row.writes == row.writes, f"step {i} writes"


@pytest.mark.slow
def test_codegen_engine_matches_generic_engine():
    """The generated straight-line engine is a dispatch change, not a
    semantics change: its traces must be identical to the generic CSR
    engine's, checkpoint rebuilds included.  Uses the nestest prefix so
    high-address fp32 reads and direct WRITEs are exercised."""
    from transformer_vm.nes.nestest import START_PC, load_image

    image = load_image()
    traces = {}
    for matmul in ("engine", "codegen"):
        machine = Cpu6502TransformerMachine(
            bcd=False, matmul=matmul, checkpoint_tokens=64
        )
        machine.boot_image(list(image), start_pc=START_PC)
        traces[matmul] = [machine.machine_step() for _ in range(60)]
        assert machine.checkpoints >= 1, matmul
    for a, b in zip(traces["engine"], traces["codegen"], strict=True):
        assert a.state == b.state
        assert a.writes == b.writes

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = START_PC
    for i, row in enumerate(traces["codegen"]):
        ref_row = ref.step()
        a, b = ref_row.state, row.state
        assert (a.pc, a.a, a.x, a.y, a.s, a.p) == (b.pc, b.a, b.x, b.y, b.s, b.p), (
            f"step {i}: ref={a} codegen={b}"
        )


@pytest.mark.slow
def test_run_loop_matches_per_step():
    """The generated C++ run loop (KernelBus shape) must produce exactly the
    trace the verified per-step machine_step path produces -- same states,
    same writes -- and detect the jump-to-self trap.  Also checks the
    chunked stream (w_row rebased per chunk) against the single native run."""
    from collections import defaultdict

    from transformer_vm.nes.nestest import START_PC, load_image

    image = load_image()
    n = 120

    # Verified reference: the per-step path (lockstep-exact vs the oracle).
    per = Cpu6502TransformerMachine(bcd=False)
    assert per.supports_run_loop
    per.boot_image(list(image), start_pc=START_PC)
    per_rows = [per.machine_step() for _ in range(n)]

    def as_tuples(out):
        wbr = defaultdict(set)
        for r, a, v in zip(out["w_row"], out["w_addr"], out["w_value"], strict=True):
            wbr[int(r)].add((int(a), int(v)))
        return [
            (
                int(out["pc"][i]), int(out["a"][i]), int(out["x"][i]),
                int(out["y"][i]), int(out["s"][i]), int(out["p"][i]),
                int(out["ir"][i]), int(out["ea"][i]), int(out["data"][i]),
                wbr.get(i, set()),
            )
            for i in range(len(out["pc"]))
        ]

    def per_tuple(row):
        s = row.state
        return (
            s.pc, s.a, s.x, s.y, s.s, s.p, s.ir, s.ea, s.data,
            {(w.addr, w.value) for w in row.writes},
        )

    single = Cpu6502TransformerMachine(bcd=False)
    out = single.run_image_arrays(bytes(image), start_pc=START_PC, max_steps=n)
    single_rows = as_tuples(out)
    assert len(single_rows) == n
    assert single_rows == [per_tuple(r) for r in per_rows]

    # Chunked stream reconstructs the identical trace across chunk boundaries.
    chunked = Cpu6502TransformerMachine(bcd=False)
    streamed = []
    for chunk in chunked.iter_run_arrays(bytes(image), start_pc=START_PC,
                                         max_steps=n, chunk=17):
        streamed.extend(as_tuples(chunk))
    assert streamed == single_rows

    # Trap detection: the verify program ends in JMP-to-self.  Every
    # executed row must still lockstep the reference oracle.
    trap = Cpu6502TransformerMachine(bcd=False)
    tout = trap.run_image_arrays(
        verify_program(), start_pc=VERIFY_START, max_steps=1000, stop_on_trap=True
    )
    assert tout["reason"] == "trap"
    ref = reference.Cpu6502(memory=bytearray(verify_program()), bcd_enabled=False)
    ref.pc = VERIFY_START
    for i in range(len(tout["pc"])):
        a = ref.step().state
        got = (int(tout["pc"][i]), int(tout["a"][i]), int(tout["x"][i]),
               int(tout["y"][i]), int(tout["s"][i]), int(tout["p"][i]))
        assert (a.pc, a.a, a.x, a.y, a.s, a.p) == got, f"trap step {i}"
