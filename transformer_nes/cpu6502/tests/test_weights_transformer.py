"""The flagship claim: the 6502 executes as real transformer weights.

Builds the analytic weights (ASAP schedule, no training) and runs genuine
transformer inference -- per-position embeddings, Q/K/V projections, hull
hardmax attention, ReGLU FFN, output head -- lockstep against the
reference core.  Expensive: real inference through the generated float32
codegen C++ engine, lockstep against the reference oracle.
"""

from __future__ import annotations

import pytest

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.weights.runner import (
    VERIFY_START,
    Cpu6502TransformerMachine,
    verify_program,
)


def lockstep_vs_reference(trace, image):
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
    image = verify_program()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=30)
    assert len(trace.rows) == 30
    lockstep_vs_reference(trace, image)


def test_region_ops_keep_write_position_in_sync():
    """inject_region advances the engine sequence one WRITE token per byte
    and map_region not at all; the Python position mirror must match both,
    or the next single-byte poke throws.  MMC2/MMC4 hit exactly this: a
    bank switch (inject_region) and the clobbered-register repair
    (inject_write) inside ONE pause servicing, with no run() between to
    resync the mirror (Punch-Out, found 2026-07-13)."""
    machine = Cpu6502TransformerMachine(bcd=False)
    machine.load_image(bytes(65536), start_pc=0x0400)
    machine.inject_region(0x8000, bytes(range(1, 65)))
    machine.inject_write(0x0010, 0x42)  # threw before the mirror fix
    machine.map_region(0x9000, bytes(range(1, 65)))
    machine.inject_write(0x0011, 0x43)
    assert machine.read_ram(0x8001) == 2
    assert machine.read_ram(0x9001) == 2
    assert machine.read_ram(0x0010) == 0x42
    assert machine.read_ram(0x0011) == 0x43


@pytest.mark.slow
def test_transformer_int32_engine_matches_high_address_nestest_prefix():
    from transformer_nes.nes.examples.nestest import START_PC, load_image

    image = load_image()
    machine = Cpu6502TransformerMachine(bcd=False)
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
def test_run_loop_matches_per_step():
    """The generated C++ run loop (EngineBus shape) must produce exactly the
    trace the verified per-step machine_step path produces -- same states,
    same writes -- and detect the jump-to-self trap.  Also checks the
    chunked stream (w_row rebased per chunk) against the single native run."""
    from collections import defaultdict

    from transformer_nes.nes.examples.nestest import START_PC, load_image

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

    from transformer_nes.cpu6502.backends.weights.windows import (
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
            assert len(match_shape_windows(image, m.engine.shapes_meta)) == 4
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


def store_loop_image() -> bytearray:
    """Three store-carrying micro-loops plus a spin: an indexed copy, an
    RMW-led counter, and a stack-push loop -- every store bounded below
    $2000 and disjoint from its window's own bytes."""
    image = bytearray(0x10000)
    image[0x0400:0x0415] = bytes([
        0xBD, 0x00, 0x06,  # 0400 LDA $0600,X  (copy loop: read leads)
        0x9D, 0x00, 0x03,  # 0403 STA $0300,X  (bound [0300, 03FF])
        0xE8,              # 0406 INX
        0xD0, 0xF7,        # 0407 BNE $0400
        0xE6, 0x20,        # 0409 INC $20      (RMW leads its window)
        0xCA,              # 040B DEX
        0xD0, 0xFB,        # 040C BNE $0409
        0x48,              # 040E PHA          (bound [0100, 01FF])
        0x88,              # 040F DEY
        0xD0, 0xFC,        # 0410 BNE $040E
        0x4C, 0x12, 0x04,  # 0412 JMP $0412 (self)
    ])
    for i in range(256):
        image[0x0600 + i] = (i * 7 + 3) & 0xFF
    return image


@pytest.mark.slow
def test_store_windows_match_per_step():
    """Store-licensed windows: the pinned module must execute the copy /
    RMW / stack-push loops as single forwards with traces (writes
    included) and final RAM byte-identical to the per-instruction path."""
    import numpy as np

    from transformer_nes.cpu6502.backends.weights.windows import (
        scan_sequence_windows,
    )

    image = store_loop_image()
    windows = scan_sequence_windows(image)
    assert {(h, r) for h, _c, r in windows} == {
        (0x0400, 1), (0x0409, 1), (0x040E, 1), (0x0412, 4),
    }

    fields = ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi",
              "ea", "data", "halted", "w_addr", "w_value")
    traces, rams = [], []
    for cfg in ({"sequence_windows": windows}, {"shape_library": ()}):
        m = Cpu6502TransformerMachine(bcd=False, **cfg)
        m.load_image(bytes(image), start_pc=0x0400)
        chunks = []
        done = 0
        while done < 2800:  # odd chunks: budgets land mid-window
            out = m.run(min(701, 2800 - done))
            assert len(out["pc"]), out["reason"]
            chunks.append(out)
            done += int(out["steps"])
        traces.append({
            k: np.concatenate([c[k] for c in chunks]) for k in fields
        })
        rams.append(m.dump_ram())
        if "sequence_windows" in cfg:
            assert m.seq_module is not None
            # The loop heads ran hundreds of times; through the module
            # they must never reach the per-instruction counter.
            hot = set(int(pc) for pc in m.engine.jit_hot(100))
            assert not hot & {0x0400, 0x0409, 0x040E}, hot
    for k in fields:
        assert np.array_equal(traces[0][k], traces[1][k]), f"{k} diverged"
    assert rams[0] == rams[1]
    assert rams[0][0x0300:0x0400] == bytes(image[0x0600:0x0700])


@pytest.mark.slow
def test_store_windows_decline_under_low_watch():
    """A write watch below $2000 must force every store window back to the
    per-instruction path, where the watched write pauses as usual: the
    static dispatch guard is what makes licensed stores safe."""
    from transformer_nes.cpu6502.backends.weights.windows import (
        scan_sequence_windows,
    )

    image = store_loop_image()
    m = Cpu6502TransformerMachine(
        bcd=False, sequence_windows=scan_sequence_windows(image),
    )
    m.load_image(bytes(image), start_pc=0x0400)
    m.watch_write(0x0020, 0x0020)
    hits = 0
    done = 0
    while done < 2800:
        out = m.run(2800 - done)
        done += int(out["steps"])
        for addr, value in zip(out["hit_addr"], out["hit_value"], strict=True):
            assert int(addr) == 0x0020
            hits += 1
            assert int(value) == hits & 0xFF
    assert hits == 256  # the INC $20 loop paused on every single write
    # The store windows really did fall back while watched.
    hot = set(int(pc) for pc in m.engine.jit_hot(100))
    assert {0x0400, 0x0409, 0x040E} <= hot, hot


@pytest.mark.slow
def test_store_window_license_rejections():
    """Unlicensable windows must be skipped, never mis-dispatched: a store
    to a register range ($2007) and a store into the window's own bytes."""
    mmio_store = (0x0500, bytes([
        0xA9, 0x55,        # LDA #$55
        0x8D, 0x07, 0x20,  # STA $2007 (>= $2000: unlicensed)
        0xCA,              # DEX
        0xD0, 0xF8,        # BNE $0500
    ]), 1)
    self_modify = (0x0300, bytes([
        0xA9, 0x55,        # LDA #$55
        0x8D, 0x05, 0x03,  # STA $0305 (inside the window's own bytes)
        0xD0, 0xF9,        # BNE $0300
    ]), 1)
    m = Cpu6502TransformerMachine(
        bcd=False, sequence_windows=(mmio_store, self_modify),
    )
    assert m.seq_module is None  # both rejected by the license


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
            assert m.engine.window_shape(0x0300) == 0
        chunks = []
        done = 0
        while done < 800:
            out = m.run(97)  # odd budgets: boundaries land mid-window
            assert len(out["pc"]), out["reason"]
            chunks.append(out)
            done += len(out["pc"])
        traces.append({k: np.concatenate([c[k] for c in chunks]) for k in fields})
        if "shape_library" not in cfg:
            assert m.engine.window_shape(0x0300) != 0, "JIT never added the window"
    n = min(len(traces[0]["pc"]), len(traces[1]["pc"]))
    for k in fields:
        assert np.array_equal(traces[0][k][:n], traces[1][k][:n]), f"{k} diverged"
