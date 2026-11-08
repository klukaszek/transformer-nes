"""The CPU chip on its native host: it never needs the reference emulator,
its loop circuits are exact, and device reads pause, resume and race
interrupts as the hardware does."""

import gc
import subprocess
import sys

import numpy as np
import pytest

from transformer_nes.cpu6502 import chip as cpu
from transformer_nes.cpu6502.runner import Cpu6502TransformerMachine


def test_neural_pipeline_does_not_import_the_reference_emulator():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'transformer_nes.cpu6502.reference':
            raise AssertionError('neural pipeline imported the reference emulator')

guard = NoReference()
sys.meta_path.insert(0, guard)
from transformer_nes.cpu6502.graph import Cpu6502Graph
from transformer_nes.cpu6502.experts import build_opcode_execution_experts
from transformer_nes.cpu6502.runner import Cpu6502TransformerMachine
from transformer_nes.nes.backends import backend_factory

graph = Cpu6502Graph(bcd=False, wide_bits=True)
assert len(build_opcode_execution_experts(graph.program_graph(), [0xA9])) == 1
machine, is_engine = backend_factory('weights')(bytes(65536), 0x8000)
assert is_engine and isinstance(machine, Cpu6502TransformerMachine)
image = bytearray(65536)
image[0x8000:0x8002] = bytes([0xA9, 0x42])
machine.load_image(image, start_pc=0x8000)
machine.run(1, trace=False)
assert machine.get_state()['a'] == 0x42

# Explicit requests for the public reference API still work.
sys.meta_path.remove(guard)
from transformer_nes.cpu6502 import Cpu6502, run
from transformer_nes.cpu6502 import reference
assert Cpu6502 is reference.Cpu6502 and run is reference.run
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_counter_loops_cover_every_licensed_state_and_limit():
    """All nonzero byte counters and byte-sized budgets, against literal
    steps: the loop circuits' new counter, N and Z, iterations and cycles."""
    from transformer_nes.nes.model import compiled

    engine = compiled().chip("cpu")
    loops = {name: f for name, f in engine.chip.constants["executions"].items() if f["kind"] == "exec_loop"}
    assert len(loops) == 4
    field = {name: i for i, name in enumerate(cpu.INPUTS)}
    out = {name: i for i, name in enumerate(cpu.STEP_OUTPUTS)}
    for name, f in loops.items():
        model = engine.model_id("step", name)
        register = "x" if f["reg"] == 1 else "y"
        for value in range(1, 256):
            path = [value]
            while (following := (path[-1] + f["delta"]) & 255) != 0:
                path.append(following)
            for limit in range(256):
                count = min(limit, len(path) - 1)
                for cycles in (5, 6):
                    inputs = np.zeros(len(cpu.INPUTS), dtype=np.int32)
                    inputs[field["is_state"]] = 1
                    inputs[field[f"in_{register}"]] = value
                    inputs[field["in_loop_limit"]] = limit
                    inputs[field["in_loop_cycles"]] = cycles
                    result = engine.step(inputs, model)
                    assert result[out[f"step_{register}"]] == path[count]
                    assert result[out["step_p_n"]] == path[count] >> 7
                    assert result[out["step_p_z"]] == 0
                    assert result[out["step_iterations"]] == count
                    assert result[out["step_cycles"]] == count * cycles


def test_machine_keeps_its_chip_alive_without_a_separate_handle():
    machine = Cpu6502TransformerMachine()
    image = bytearray(65536)
    image[0x8000:0x8003] = bytes([0xA9, 0x42, 0xEA])
    machine.load_image(image, start_pc=0x8000)
    gc.collect()
    machine.run(1, trace=False)
    assert machine.get_state()["a"] == 0x42


@pytest.mark.parametrize("union_only", [False, True])
def test_read_requests_pause_without_committing_then_consume_the_reply(union_only):
    from transformer_nes.cpu6502.reference import Cpu6502, instruction_cycles
    from transformer_nes.cpu6502.spec import MEMORY_MODES, OPCODES, READS_MEMORY

    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    for byte, op in OPCODES.items():
        if op.operation not in READS_MEMORY or op.mode not in MEMORY_MODES:
            continue
        for index in (0, 1, 127, 255):
            image = bytearray([0xA7]) * 65536
            image[0x8000:0x8003] = bytes([byte, 255, 255])
            # Both indirect modes exercise zero-page wrapping; indexed
            # reads also cross pages and the top of the address space.
            pointer = ((255 + index) & 255) if op.mode == "idx" else 255
            image[pointer] = 255
            image[(pointer + 1) & 255] = 0x40
            ref = Cpu6502(memory=image.copy(), bcd_enabled=False)
            ref.pc, ref.a, ref.x, ref.y = 0x8000, 0x63, index, index
            address = ref.resolve_ea(op, 255, 255)
            cycles = instruction_cycles(ref, op, ref.pc, ref.x, ref.y, ref.p)
            machine.load_image(image, start_pc=ref.pc)
            machine.engine.set_run_state(
                ref.pc, ref.a, ref.x, ref.y, ref.s, ref.p, 0, 0, 0, 0, 0, 0
            )
            machine.watch_read(0, 65535)
            before = machine.get_state()
            request = machine.run(1, trace=False)
            assert request["reason"] == "watch_read"
            assert (request["read_addr"], request["read_cycles"]) == (address, cycles)
            assert machine.get_state() == before
            assert machine.dump_ram() == bytes(image)
            # The body must use this fresh reply rather than a value read
            # speculatively while constructing the request packet.
            ref.write_memory(address, 0x52)
            machine.inject_write(address, 0x52)
            ref.step()
            result = machine.run(1, trace=False)
            after = machine.get_state()
            assert result["cycles"] == cycles
            assert tuple(after[k] for k in ("pc", "a", "x", "y", "s", "p")) == (
                ref.pc,
                ref.a,
                ref.x,
                ref.y,
                ref.s,
                ref.p,
            ), (byte, index)
            assert machine.dump_ram() == bytes(ref.memory)


@pytest.mark.parametrize("union_only", [False, True])
def test_read_continuation_retains_pointer_and_instruction_context(union_only):
    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    image = bytearray(65536)
    image[0x8000:0x8002] = bytes([0xB1, 0x20])  # LDA ($20),Y
    image[0x20:0x22] = bytes([0xFF, 0x40])
    machine.load_image(image, start_pc=0x8000)
    machine.engine.set_run_state(0x8000, 0, 0, 1, 0xFD, 0x24, 0, 0, 0, 0, 0, 0)
    machine.watch_read(0x4100, 0x4100)
    request = machine.run(1, trace=False)
    assert (request["read_addr"], request["read_cycles"]) == (0x4100, 6)
    # A device reply may alter mapped memory. Neither its changed pointer
    # nor a replacement opcode may change the instruction already begun.
    for addr, value in (
        (0x20, 0),
        (0x21, 0x60),
        (0x8000, 0xEE),
        (0x8001, 0x30),
        (0x8002, 0x60),
        (0x4100, 0x52),
    ):
        machine.inject_write(addr, value)
    result = machine.run(1, trace=True)
    state = machine.get_state()
    assert (state["pc"], state["a"], result["cycles"]) == (0x8002, 0x52, 6)
    assert (result["ir"][0], result["oplo"][0], result["ea"][0]) == (0xB1, 0x20, 0x4100)
    assert machine.read_ram(0x4100) == 0x52


@pytest.mark.parametrize("union_only", [False, True])
@pytest.mark.parametrize("trace", [False, True])
def test_read_continuation_rejects_a_changed_shape_prefix(union_only, trace):
    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    image = bytearray(65536)
    image[0x8000:0x8007] = bytes([0xAD, 0, 0x40, 0x29, 0x80, 0xF0, 0xF9])
    machine.load_image(image, start_pc=0x8000)
    machine.watch_read(0x4000, 0x4000)
    assert machine.run(3, trace=False)["reason"] == "watch_read"
    machine.inject_write(0x4000, 0x80)
    machine.inject_write(0x8001, 1)  # new $4001 must not replace the latched $4000
    out = machine.run(3, trace=trace)
    state = machine.get_state()
    assert (state["pc"], state["a"], out["cycles"]) == (0x8007, 0x80, 8)
    if trace:
        assert (out["oplo"][0], out["ophi"][0], out["ea"][0]) == (0, 0x40, 0x4000)


@pytest.mark.parametrize("union_only", [False, True])
@pytest.mark.parametrize("interrupt", ["nmi", "irq"])
def test_read_continuation_finishes_before_an_interrupt_raised_by_the_reply(union_only, interrupt):
    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    image = bytearray(65536)
    image[0x8000:0x8004] = bytes([0xAD, 0, 0x40, 0xEA])
    image[0xFFFA:0xFFFC] = image[0xFFFE:0x10000] = bytes([0, 0x90])
    machine.load_image(image, start_pc=0x8000)
    machine.engine.set_run_state(0x8000, 0, 0, 0, 0xFD, 0x20, 0, 0, 0, 0, 0, 0)
    machine.watch_read(0x4000, 0x4000)
    assert machine.run(1, trace=False)["reason"] == "watch_read"
    getattr(machine, f"set_{interrupt}")()
    machine.inject_write(0x4000, 0x52)
    assert machine.run(1, trace=False)["cycles"] == 4
    assert (machine.get_state()["pc"], machine.get_state()["a"]) == (0x8003, 0x52)
    assert machine.run(1, trace=False)["cycles"] == 7
    assert machine.get_state()["pc"] == 0x9000
    assert (machine.read_ram(0x1FD), machine.read_ram(0x1FC)) == (0x80, 3)


@pytest.mark.parametrize("opcode", [0xAD, 0x8D])
def test_watched_address_unions_include_boundaries_and_clear_on_reload(opcode):
    machine = Cpu6502TransformerMachine(bcd=False)
    cases = [((0, 0),), ((63, 64),), ((64, 130),), ((65535, 65535),),
             ((0, 65535),), ((1, 62), (65, 127), (61, 67)), ((130, 64),)]
    for ranges in cases:
        addresses = {0, 1, 62, 63, 64, 65, 127, 128, 130, 131, 65534, 65535}
        for lo, hi in ranges:
            addresses.update(a for a in (lo-1, lo, hi, hi+1) if 0 <= a <= 65535)
        for address in addresses:
            for clear in (False, True):
                image = bytearray(65536)
                image[0x8000:0x8003] = bytes((opcode, address & 255, address >> 8))
                machine.load_image(image, start_pc=0x8000)
                for lo, hi in ranges:
                    (machine.watch_read if opcode == 0xAD else machine.watch_write)(lo, hi)
                if clear:
                    machine.load_image(image, start_pc=0x8000)
                result = machine.run(1, trace=False)
                watched = not clear and any(lo <= address <= hi for lo, hi in ranges)
                if opcode == 0xAD:
                    assert (result["reason"] == "watch_read") == watched
                    if watched:
                        assert result["read_addr"] == address
                else:
                    assert result["hit_addr"].tolist() == ([address] if watched else [])
