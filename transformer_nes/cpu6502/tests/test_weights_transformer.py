"""The flagship claim: the 6502 executes as real transformer weights.

Builds the analytic weights (ASAP schedule, no training) and runs genuine
transformer inference -- per-position embeddings, Q/K/V projections, and
licensed exact attention state
attention, ReGLU FFN, output head -- lockstep against the
reference core, through the compiled int32 Torch bank.
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


@pytest.mark.parametrize("union_only", [False, True])
def test_write_effects_carry_rmw_operands_without_host_opcode_decoding(union_only):
    from transformer_nes.cpu6502.spec import MODE_ACCUMULATOR, OPCODES

    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    machine.engine.shapes_meta = []
    rmw_operations = {"inc", "dec", "asl", "lsr", "rol", "ror", "dcp", "isb", "slo", "rla", "sre", "rra"}
    for op in OPCODES.values():
        for value in (0, 1, 127, 128, 254, 255):
            image = bytearray(65536)
            image[0x0400:0x0403] = bytes((op.byte, 0x20, 0x60))
            ref = reference.Cpu6502(memory=image, bcd_enabled=False)
            ref.pc, ref.a, ref.x, ref.y, ref.p = 0x0400, 0xA5, 3, 5, 0x24 | (value & 1)
            lo, hi, _ = ref.read_operand(op)
            address = ref.resolve_ea(op, lo, hi)
            if op.mode != MODE_ACCUMULATOR:
                ref.write_memory(address, value)
            machine.load_image(bytes(ref.memory), start_pc=ref.pc)
            machine.engine.set_run_state(ref.pc, ref.a, ref.x, ref.y, ref.s, ref.p, 0, 0, 0, 0, 0, 0)
            machine.watch_write(0, 65535)
            expected = ref.step().writes
            actual = machine.run(1, trace=bool(value & 1))
            receipt = list(zip(actual["hit_addr"], actual["hit_value"], actual["hit_original"], strict=True))
            assert receipt == [(w.addr, w.value, w.original) for w in expected], (op, value)
            rmw = op.operation in rmw_operations and op.mode != MODE_ACCUMULATOR
            assert [original for _, _, original in receipt] == ([value] if rmw else [-1] * len(receipt))
            if value & 1:
                assert actual["w_original"].tolist() == [w.original for w in expected]


@pytest.mark.parametrize("union_only", [False, True])
@pytest.mark.parametrize("trace", [False, True])
def test_rmw_receipt_uses_the_serviced_reply_after_code_changes(union_only, trace):
    image = bytearray(65536)
    image[0x0400:0x0403] = bytes.fromhex("ee00d0")  # INC $D000
    image[0xFFFA:0xFFFC] = bytes.fromhex("0005")
    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    machine.load_image(image, start_pc=0x0400)
    machine.watch_read(0xD000, 0xD000)
    machine.watch_write(0xD000, 0xD000)
    request = machine.run(1, trace=trace)
    assert request["reason"] == "watch_read" and not len(request["hit_original"])
    # The device reply changes code and raises NMI before completion.
    machine.inject_write(0x0400, 0x8D)  # memory now says STA, not INC
    machine.inject_write(0xD000, 0xFE)
    machine.set_nmi()
    out = machine.run(1, trace=trace)
    assert out["hit_original"].tolist() == [0xFE]
    assert out["hit_value"].tolist() == [0xFF]
    assert machine.read_ram(0xD000) == 0xFF and machine.get_state()["pc"] == 0x0403
    machine.watch_write(0x0100, 0x01FF)
    interrupt = machine.run(1, trace=trace)
    assert interrupt["hit_original"].tolist() == [-1, -1, -1]
    assert machine.get_state()["pc"] == 0x0500


@pytest.mark.slow
def test_transformer_weights_match_reference():
    machine = Cpu6502TransformerMachine(bcd=False)
    image = verify_program()
    assert len(image) == 0x10000
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=30)
    assert len(trace.rows) == 30
    lockstep_vs_reference(trace, image)


def test_region_ops_keep_write_position_in_sync():
    """Mapping a bank adds no WRITE tokens; a later injected write must
    still agree with the engine's sequence position."""
    machine = Cpu6502TransformerMachine(bcd=False)
    machine.load_image(bytes(65536), start_pc=0x0400)
    position = machine.get_state()["pos"]
    machine.map_region(0x8000, bytes(range(1, 65)))
    machine.inject_write(0x0010, 0x42)
    machine.map_region(0x9000, bytes(range(1, 65)))
    machine.inject_write(0x0011, 0x43)
    assert machine.read_ram(0x8001) == 2
    assert machine.read_ram(0x9001) == 2
    assert machine.read_ram(0x0010) == 0x42
    assert machine.read_ram(0x0011) == 0x43

    assert machine.get_state()["pos"] == position + 2


@pytest.mark.slow
def test_transformer_int32_engine_matches_high_address_nestest_prefix():
    from transformer_nes.cpu6502.examples.nestest import START_PC, load_image

    image = load_image()
    machine = Cpu6502TransformerMachine(bcd=False, union_only=True)
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
@pytest.mark.parametrize("union_only", [False, True])
def test_the_model_times_every_instruction(union_only):
    """Elapsed time is an output of the forward, like the registers: each
    nestest instruction's cycles, page-crossing branches and indexed reads
    included, from the opcode experts and from the union model alike."""
    from transformer_nes.cpu6502.examples.nestest import START_PC, load_image, new_cpu

    image = load_image()
    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    machine.load_image(bytes(image), start_pc=START_PC)
    ref = new_cpu(image)
    for step in range(8991):
        op = reference.opcode_for(ref.read_memory(ref.pc))
        expected = reference.instruction_cycles(ref, op, ref.pc, ref.x, ref.y, ref.p)
        got = int(machine.run(1, max_cycles=0, trace=False)["cycles"])
        assert got == expected, f"step {step} at ${ref.pc:04X}: {op.mnemonic} {op.mode}"
        ref.step()


@pytest.mark.slow
def test_run_loop_and_chunks_match_union_steps():
    """Routed and chunked execution match one-instruction union forwards,
    including writes and jump-to-self trap detection."""
    from collections import defaultdict

    from transformer_nes.cpu6502.examples.nestest import START_PC, load_image

    image = load_image()
    n = 120

    def as_tuples(out):
        wbr = defaultdict(set)
        for r, a, v in zip(out["w_row"], out["w_addr"], out["w_value"], strict=True):
            wbr[int(r)].add((int(a), int(v)))
        return [
            (
                int(out["pc"][i]),
                int(out["a"][i]),
                int(out["x"][i]),
                int(out["y"][i]),
                int(out["s"][i]),
                int(out["p"][i]),
                int(out["ir"][i]),
                int(out["ea"][i]),
                int(out["data"][i]),
                wbr.get(i, set()),
            )
            for i in range(len(out["pc"]))
        ]

    per = Cpu6502TransformerMachine(bcd=False, union_only=True)
    per.load_image(image, start_pc=START_PC)
    per_rows = []
    for step in range(n):
        out = per.run(1)
        out["w_row"] -= step
        per_rows.extend(as_tuples(out))

    single = Cpu6502TransformerMachine(bcd=False)
    out = single.run_image_arrays(bytes(image), start_pc=START_PC, max_steps=n)
    single_rows = as_tuples(out)
    assert len(single_rows) == n
    assert single_rows == per_rows

    # Chunked stream reconstructs the identical trace across chunk boundaries.
    chunked = Cpu6502TransformerMachine(bcd=False)
    streamed = []
    for chunk in chunked.iter_run_arrays(bytes(image), start_pc=START_PC, max_steps=n, chunk=17):
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
        got = (
            int(tout["pc"][i]),
            int(tout["a"][i]),
            int(tout["x"][i]),
            int(tout["y"][i]),
            int(tout["s"][i]),
            int(tout["p"][i]),
        )
        assert (a.pc, a.a, a.x, a.y, a.s, a.p) == got, f"trap step {i}"


@pytest.mark.slow
def test_fixed_shapes_match_per_step():
    """Traced fallback stays byte-identical across configurations; untraced
    pinned/shape execution models finish with the same state, RAM and cycles
    as the per-instruction path across polls, branches and odd boundaries."""
    import numpy as np

    from transformer_nes.cpu6502.backends.weights.shapes import (
        match_shape_windows,
    )

    image = bytearray(0x10000)
    image[0x0400:0x0413] = bytes(
        [
            0xAD,
            0x00,
            0x07,  # 0400 LDA $0700   (abs poll trio)
            0x29,
            0x40,  # 0403 AND #$40
            0xF0,
            0xF9,  # 0405 BEQ $0400
            0xEA,  # 0407 NOP
            0xCA,  # 0408 DEX         (delay loop)
            0xD0,
            0xFD,  # 0409 BNE $0408
            0xA5,
            0x10,  # 040B LDA $10     (zero-page BPL poll)
            0x10,
            0xFC,  # 040D BPL $040B
            0xEA,  # 040F NOP
            0x4C,
            0x10,
            0x04,  # 0410 JMP $0410 (self)
        ]
    )
    fields = ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi", "ea", "data", "halted")
    modes = ("shape", "compact")
    traces = []
    for mode in modes:
        m = Cpu6502TransformerMachine(bcd=False)
        m.load_image(bytes(image), start_pc=0x0400)
        if mode == "compact":
            m.engine.clear_windows()
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

        drain(3000)  # abs poll trio spins
        m.inject_write(0x0700, 0x40)  # flag up: through NOP + DEX/BNE delay
        drain(1000)  # ...into the zp BPL poll
        m.inject_write(0x0010, 0x80)  # bit 7 up: fall through to the spin
        drain(1000)
        traces.append({k: np.concatenate([c[k] for c in chunks]) for k in fields})
    for t in traces[1:]:
        for k in fields:
            assert np.array_equal(traces[0][k], t[k]), f"{k} diverged"
    for pc in (0x0408, 0x040B, 0x0410):
        assert pc in traces[0]["pc"], f"never reached {pc:04X}"

    # Gameplay does not request instruction traces.  In that mode the
    # sequence models use the compact execution receipt; compare persistent
    # state/RAM/cycles after the same boundary and input schedule.
    final = []
    for mode in modes:
        m = Cpu6502TransformerMachine(bcd=False)
        m.load_image(bytes(image), start_pc=0x0400)
        if mode == "compact":
            m.engine.clear_windows()

        def run_untraced(n, m=m):
            done = cycles = 0
            while done < n:
                out = m.run(min(701, n - done), trace=False)
                assert int(out["steps"]) > 0, out["reason"]
                done += int(out["steps"])
                cycles += int(out["cycles"])
            assert done == n
            return cycles

        cycles = run_untraced(3000)
        m.inject_write(0x0700, 0x40)
        cycles += run_untraced(1000)
        m.inject_write(0x0010, 0x80)
        cycles += run_untraced(1000)
        state = m.get_state()
        state = {
            name: state[name]
            for name in ("pc", "a", "x", "y", "s", "p", "halted", "steps", "cycles")
        }
        final.append((state, m.dump_ram(), cycles))
    assert final[0] == final[1]


@pytest.mark.slow
def test_compact_stack_and_control_match_reference():
    image = bytearray(0x10000)
    image[0x0400:0x0405] = bytes(
        [
            0xA2,
            0xFC,  # LDX #$FC
            0x9A,  # TXS
            0x40,  # RTI -> $0410 from the prepared stack
            0xEA,
        ]
    )
    image[0x0410:0x0425] = bytes(
        [
            0xA9,
            0x42,  # LDA #$42
            0x48,  # PHA
            0xA9,
            0x00,  # LDA #$00
            0x68,  # PLA
            0x08,  # PHP
            0x18,  # CLC
            0x28,  # PLP
            0x20,
            0x50,
            0x04,  # JSR $0450
            0x4C,
            0x22,
            0x04,  # JMP $0422
            0xEA,
            0xEA,
            0xEA,
            0x6C,
            0x50,
            0x00,  # JMP ($0050) -> $0430
        ]
    )
    image[0x0430:0x0433] = bytes([0x00, 0xEA, 0x00])
    image[0x0450:0x0452] = bytes([0xE8, 0x60])  # INX; RTS
    image[0x0460:0x0469] = bytes(
        [
            0xA9,
            0x00,
            0x8D,
            0xFE,
            0xFF,
            0x8D,
            0xFF,
            0xFF,
            0x40,
        ]
    )
    image[0x0050:0x0052] = bytes([0x30, 0x04])
    image[0x01FD:0x0200] = bytes([0xA5, 0x10, 0x04])
    image[0xFFFE:0x10000] = bytes([0x60, 0x04])

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = 0x0400
    ref_rows = []
    while not ref.halted:
        ref_rows.append(ref.step())
    assert len(ref_rows) == 21

    machine = Cpu6502TransformerMachine(bcd=False)
    machine.load_image(bytes(image), start_pc=0x0400)
    out = machine.run(100, trace=False)

    assert out["reason"] == "halt"
    assert int(out["steps"]) == len(ref_rows)
    state = machine.get_state()
    assert tuple(state[name] for name in ("pc", "a", "x", "y", "s", "p", "halted")) == (
        ref.pc,
        ref.a,
        ref.x,
        ref.y,
        ref.s,
        ref.p,
        ref.halted,
    )
    assert machine.dump_ram() == bytes(ref.memory)


def test_neural_next_pc_covers_branch_and_address_wrap_boundaries():
    """Untraced experts decide control even when operands and targets wrap.

    NOP also needs a forward: advancing its PC is neural work, despite
    preserving every byte register and flag.
    """
    machine = Cpu6502TransformerMachine()
    machine.engine.shapes_meta = []
    cases = [
        (head, bytes([opcode, offset]), flags)
        for opcode in (0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0)
        for flags in (0x24, 0xE7)
        for head in (0, 0x00FE, 0x7FFF, 0xFFFE, 0xFFFF)
        for offset in (0, 1, 0x7F, 0x80, 0xFE, 0xFF)
    ]
    cases += [
        (head, code, 0x24)
        for head in (0xFFFE, 0xFFFF)
        for code in (b"\xea", b"\xa9\x42", b"\x4c\xff\xff", b"\x4c\x00\x00")
    ]
    calls_before = machine._torch_forwards.counts[0]
    for head, code, flags in cases:
        image = bytearray(65536)
        for offset, byte in enumerate(code):
            image[(head + offset) & 0xFFFF] = byte
        ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
        ref.pc, ref.p = head, flags
        cycles = reference.instruction_cycles(
            ref, reference.opcode_for(code[0]), head, ref.x, ref.y, flags
        )
        ref.step()
        machine.load_image(image, start_pc=head)
        machine.engine.set_run_state(head, 0, 0, 0, 0xFD, flags, 0, 0, 0, 0, 0, 0)
        result = machine.run(1, trace=False)
        actual = machine.get_state()
        assert (actual["pc"], actual["a"], actual["p"], result["cycles"]) == (
            ref.pc, ref.a, ref.p, cycles
        ), (head, code.hex(), flags)
    assert machine._torch_forwards.counts == (calls_before + len(cases), 0)


@pytest.mark.slow
def test_full_trace_interrupt_uses_union_receipt():
    image = bytearray(0x10000)
    image[0x0400:0x0403] = bytes([0xA9, 0x55, 0xEA])
    image[0x0500] = 0xEA
    image[0xFFFA:0xFFFC] = bytes([0x00, 0x05])

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = 0x0400
    ref.nmi_pending = True
    expected = ref.step()

    machine = Cpu6502TransformerMachine(bcd=False)
    machine.load_image(bytes(image), start_pc=0x0400)
    machine.set_nmi()
    out = machine.run(1, trace=True)

    state = expected.state
    assert tuple(
        int(out[name][0])
        for name in (
            "pc",
            "a",
            "x",
            "y",
            "s",
            "p",
            "ir",
            "oplo",
            "ophi",
            "ea",
            "data",
        )
    ) == (
        state.pc,
        state.a,
        state.x,
        state.y,
        state.s,
        state.p,
        state.ir,
        state.operand_lo,
        state.operand_hi,
        state.ea,
        state.data,
    )
    assert list(zip(out["w_addr"], out["w_value"], strict=True)) == [
        (write.addr, write.value) for write in expected.writes
    ]
    assert out["w_original"].tolist() == [-1, -1, -1]


@pytest.mark.slow
def test_loop_recurrences_are_single_steps_exactly():
    """Counting and polling loops retire many iterations per dispatch at
    the model's own per-iteration cycles; under every budget the machine
    stops exactly where single steps would, in the same state."""
    code = bytes(
        [
            0xA2,
            0xC8,  # $8000 LDX #200
            0xCA,
            0xD0,
            0xFD,  # $8002 DEX; BNE $8002   (count loop)
            0xA0,
            0x07,  # $8005 LDY #7
            0x88,
            0xD0,
            0xFD,  # $8007 DEY; BNE $8007
            0xA5,
            0x10,  # $800A LDA $10           (poll: $10 never changes)
            0x10,
            0xFC,  # $800C BPL $800A
        ]
    )
    image = bytearray(0x10000)
    image[0x8000 : 0x8000 + len(code)] = code
    for budget in (1, 2, 3, 5, 100, 401, 402, 403, 604, 605, 1000, 1001, 1002, 7777):
        m = Cpu6502TransformerMachine(bcd=False)
        m.load_image(bytes(image), start_pc=0x8000)
        m.map_region(0x8000, code)  # registers the ROM's shape windows
        m.engine.engine.xray_take()
        out = m.run(100_000, max_cycles=budget, trace=False)
        _paths, _ops, dispatches = m.engine.engine.xray_take()

        ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
        ref.pc = 0x8000
        steps = cycles = 0
        while cycles < budget:
            op = reference.opcode_for(ref.read_memory(ref.pc))
            cycles += reference.instruction_cycles(ref, op, ref.pc, ref.x, ref.y, ref.p)
            ref.step()
            steps += 1
        state = m.get_state()
        assert (int(out["steps"]), int(out["cycles"])) == (steps, cycles), budget
        assert (state["pc"], state["a"], state["x"], state["y"], state["p"]) == (
            ref.pc,
            ref.a,
            ref.x,
            ref.y,
            ref.p,
        ), budget
        if budget >= 1000:  # the loops ran as recurrences, not step by step
            assert sum(dispatches) < steps / 10, (budget, steps, dispatches)


def test_parallel_flag_readouts_match_reference_for_every_byte():
    """Wrap boundaries and shifted-out bits must preserve all architectural flags."""
    machine = Cpu6502TransformerMachine()
    machine.engine.shapes_meta = []
    machine.load_image(bytes(65536), start_pc=0x8000)
    ref = reference.Cpu6502(memory=bytearray(65536), bcd_enabled=False)
    cases = [bytes([opcode]) for opcode in (0xE8, 0xC8, 0xCA, 0x88, 0x0A, 0x2A, 0x4A, 0x6A)]
    cases += [bytes([opcode, 0x20]) for opcode in (0xE6, 0xC6, 0x06, 0x26, 0x46, 0x66)]
    cases += [bytes([opcode, operand]) for opcode in (0x69, 0xE9) for operand in (0, 127, 128, 255)]
    for code in cases:
        machine.map_region(0x8000, code)
        ref.memory[0x8000 : 0x8000 + len(code)] = code
        op = reference.opcode_for(code[0])
        for value in range(256):
            for carry in (0, 1):
                flags = 0x6C | carry
                machine.map_region(0x20, bytes([value]))
                machine.engine.set_run_state(
                    0x8000, value, value, value, 0xFD, flags, 0, 0, 0, 0, 0, 0
                )
                ref.pc, ref.a, ref.x, ref.y, ref.s, ref.p = 0x8000, value, value, value, 0xFD, flags
                ref.memory[0x20] = value
                cycles = reference.instruction_cycles(ref, op, ref.pc, ref.x, ref.y, ref.p)
                ref.step()
                out = machine.run(1, trace=False)
                actual = machine.get_state()
                assert tuple(actual[key] for key in ("pc", "a", "x", "y", "s", "p")) == (
                    ref.pc,
                    ref.a,
                    ref.x,
                    ref.y,
                    ref.s,
                    ref.p,
                ), (code.hex(), value, carry)
                assert int(out["cycles"]) == cycles
                assert machine.read_ram(0x20) == ref.memory[0x20]
    assert machine._torch_forwards.counts[1] == 0


def test_counter_recurrences_cover_byte_wrap_and_page_cross_budgets():
    """Closed-form skips stop before zero, budgets and a branch page crossing."""
    machine = Cpu6502TransformerMachine()
    for opcode, register in ((0xCA, "x"), (0x88, "y"), (0xE8, "x"), (0xC8, "y")):
        for head in (0x8000, 0x80FE):
            image = bytearray(65536)
            # Counter; BNE head; NOP; JMP self.
            target = head + 4
            image[head : head + 7] = bytes(
                [opcode, 0xD0, 0xFD, 0xEA, 0x4C, target & 255, target >> 8]
            )
            machine.load_image(image, start_pc=head)
            for value in range(256):
                for budget in (1, 4, 7, 13, 2048):
                    x, y = (value, 37) if register == "x" else (37, value)
                    machine.engine.set_run_state(head, 19, x, y, 0xFD, 0xE5, 0, 0, 0, 0, 0, 0)
                    result = machine.run(2000, max_cycles=budget, trace=False)
                    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
                    ref.pc, ref.a, ref.x, ref.y, ref.p = head, 19, x, y, 0xE5
                    cycles = steps = 0
                    while cycles < budget and steps < 2000:
                        op = reference.opcode_for(ref.read_memory(ref.pc))
                        cycles += reference.instruction_cycles(ref, op, ref.pc, ref.x, ref.y, ref.p)
                        ref.step()
                        steps += 1
                    state = machine.get_state()
                    assert (state["pc"], state["a"], state["x"], state["y"], state["p"]) == (
                        ref.pc,
                        ref.a,
                        ref.x,
                        ref.y,
                        ref.p,
                    ), (opcode, head, value, budget)
                    assert (result["steps"], result["cycles"]) == (steps, cycles)


def test_cold_build_releases_reference_weights_and_still_executes(monkeypatch):
    import gc
    import weakref

    from transformer_nes.cpu6502.backends.weights import runner

    models = []
    build_model = runner.build_model

    def track_model(*args, **kwargs):
        result = build_model(*args, **kwargs)
        models.append(weakref.ref(result[0]))
        return result

    monkeypatch.setattr(runner, "build_model", track_model)
    monkeypatch.setattr(runner.GeneratedWeightsEngine, "from_cache", staticmethod(lambda key: None))
    machine = runner.Cpu6502TransformerMachine(bcd=False)
    gc.collect()
    assert models and all(model() is None for model in models)

    image = bytearray(0x10000)
    image[0x8000:0x8003] = bytes([0xA9, 0x42, 0x00])  # LDA #$42; BRK
    machine.load_image(image, start_pc=0x8000)
    machine.run(2, max_cycles=0, trace=False)
    assert machine.get_state()["a"] == 0x42
