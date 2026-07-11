"""The flagship claim: the 6502 executes as real transformer weights.

Builds the analytic weights (ASAP schedule, no training) and runs genuine
transformer inference -- per-position embeddings, Q/K/V projections, hull
hardmax attention, ReGLU FFN, output head -- lockstep against the
reference core.  Expensive: real inference through the generated float32
codegen C++ engine, lockstep against the reference oracle.
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


@pytest.mark.slow
def test_sequence_experts_match_per_step():
    """All three dispatch tiers must produce byte-identical run-loop traces:
    the per-ROM pinned module, the canonical shape table (zero per-ROM
    compile -- every idiom below matches a library shape), and the pure
    per-instruction path, across polls, mid-run flag flips, the branch
    fall-through, the spin, and chunk boundaries landing mid-window."""
    import numpy as np

    from transformer_vm.cpu6502.weights_runner import (
        match_shape_windows,
        scan_sequence_windows,
    )

    image = bytearray(0x10000)
    image[0x0400:0x0413] = bytes([
        0xAD, 0x00, 0x07,  # 0400 LDA $0700   (abs poll trio)
        0x29, 0x40,        # 0403 AND #$40
        0xF0, 0xF9,        # 0405 BEQ $0400
        0xEA,              # 0407 NOP
        0xCA,              # 0408 DEX         (delay loop)
        0xD0, 0xFD,        # 0409 BNE $0408
        0xA5, 0x10,        # 040B LDA $10     (zero-page BPL poll)
        0x10, 0xFC,        # 040D BPL $040B
        0xEA,              # 040F NOP
        0x4C, 0x10, 0x04,  # 0410 JMP $0410 (self)
    ])
    windows = scan_sequence_windows(image)
    assert {(h, r) for h, _c, r in windows} == {
        (0x0400, 1), (0x0408, 1), (0x040B, 1), (0x0410, 4),
    }

    fields = ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi",
              "ea", "data", "halted")
    configs = (
        {"sequence_windows": windows},           # pinned module + shape table
        {},                                      # shape table only
        {"shape_library": ()},                   # pure per-instruction
    )
    traces = []
    for cfg in configs:
        m = Cpu6502TransformerMachine(bcd=False, **cfg)
        m.load_image(bytes(image), start_pc=0x0400)
        if "shape_library" not in cfg:
            # Every idiom in the program must be covered by a library shape.
            assert len(match_shape_windows(image, m._engine.shapes_meta)) == 4
        chunks = []

        def drain(n, chunks=chunks, m=m):
            done = 0
            while done < n:  # odd chunks: budget boundaries land mid-window
                out = m.run(min(701, n - done))
                assert len(out["pc"]), out["reason"]
                chunks.append(out)
                done += len(out["pc"])

        drain(3000)                    # abs poll trio spins
        m.inject_write(0x0700, 0x40)   # flag up: through NOP + DEX/BNE delay
        drain(1000)                    # ...into the zp BPL poll
        m.inject_write(0x0010, 0x80)   # bit 7 up: fall through to the spin
        drain(1000)
        traces.append({
            k: np.concatenate([c[k] for c in chunks]) for k in fields
        })
    for t in traces[1:]:
        for k in fields:
            assert np.array_equal(traces[0][k], t[k]), f"{k} diverged"
    for pc in (0x0408, 0x040B, 0x0410):
        assert pc in traces[0]["pc"], f"never reached {pc:04X}"


@pytest.mark.slow
def test_jit_discovers_runtime_window():
    """The JIT counter must discover a loop the program WRITES INTO RAM at
    runtime (invisible to the boot scan), add it to the window table
    mid-run, and leave the trace byte-identical to the per-instruction
    machine across the hot-swap."""
    import numpy as np

    image = bytearray(0x10000)
    # Copy "DEX / BNE -3 / JMP $0303 (self)" to $0300, set X, jump there.
    code = bytes([0xCA, 0xD0, 0xFD, 0x4C, 0x03, 0x03])
    prog = bytearray()
    for i, b in enumerate(code):
        prog += bytes([0xA9, b, 0x8D, i, 0x03])   # LDA #b / STA $030i
    prog += bytes([0xA2, 0xFF,                    # LDX #$FF
                   0x4C, 0x00, 0x03])             # JMP $0300
    image[0x0400:0x0400 + len(prog)] = prog

    fields = ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi",
              "ea", "data", "halted")
    traces = []
    for cfg in ({"jit_interval": 1, "jit_min_count": 64},
                {"shape_library": ()}):
        m = Cpu6502TransformerMachine(bcd=False, **cfg)
        m.load_image(bytes(image), start_pc=0x0400)
        if "shape_library" not in cfg:
            # The loop does not exist at boot; only the JIT can find it.
            assert m._engine.window_shape(0x0300) == 0
        chunks = []
        done = 0
        while done < 800:
            out = m.run(97)  # odd budgets: boundaries land mid-window
            assert len(out["pc"]), out["reason"]
            chunks.append(out)
            done += len(out["pc"])
        traces.append({k: np.concatenate([c[k] for c in chunks]) for k in fields})
        if "shape_library" not in cfg:
            assert m._engine.window_shape(0x0300) != 0, "JIT never added the window"
    n = min(len(traces[0]["pc"]), len(traces[1]["pc"]))
    for k in fields:
        assert np.array_equal(traces[0][k][:n], traces[1][k][:n]), f"{k} diverged"
