"""Compiled transformer weights match eager layers and own no runtime tensor pool."""

import collections
import ctypes
import gc
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from transformer_nes.cpu6502.backends.weights.expert_runner import (
    HALVES,
    WEIGHTS,
    attention_read,
    indexed_read,
    layer,
    share_queries,
)
from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine
from transformer_nes.cpu6502.backends.weights.torch_forward import bank_module
from transformer_nes.model.torch_codegen import weight_module
from transformer_nes.model.torch_layout import layer_arrays


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
        if fullname == 'transformer_nes.cpu6502.backends.reference':
            raise AssertionError('neural pipeline imported the reference emulator')

guard = NoReference()
sys.meta_path.insert(0, guard)
from transformer_nes.cpu6502.construction import Cpu6502Graph
from transformer_nes.cpu6502.backends.weights.experts import build_opcode_execution_experts
from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine
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
from transformer_nes.cpu6502.backends import reference
assert Cpu6502 is reference.Cpu6502 and run is reference.run
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cached_bank_identity_does_not_import_the_compiler():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
from transformer_nes.cpu6502.backends.weights import torch_forward
class NoCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('torch._inductor', 'torch._dynamo')):
            raise AssertionError(f'cache lookup imported the compiler: {fullname}')
sys.meta_path.insert(0, NoCompiler())
assert torch_forward.bank_path(False).endswith('.so')
assert torch_forward.bank_path(False) != torch_forward.bank_path(True)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("read", [indexed_read, attention_read])
def test_shared_queries_preserve_byte_bits_addresses_and_half_rounding(read):
    """Many value heads share a query; differently scaled queries stay distinct."""
    pair = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.int32)
    original = torch.cat([pair] * 5 + [2 * pair, torch.tensor([[0, 0, 1]], dtype=torch.int32)])
    kinds = torch.tensor([[-1, -2], [0, 1], [2, 3], [4, 5], [6, -3], [-2, -1]])
    memory = torch.arange(256, dtype=torch.uint8)
    w_out = torch.eye(13, dtype=torch.int32)
    w_out[-1, -1] = 0
    halves = torch.zeros_like(w_out)
    halves[-1, -1] = 1
    projection, doubled, routes = share_queries(original, 2 * w_out + halves, kinds)
    w_out, halves = doubled >> 1, doubled - 2 * (doubled >> 1)
    assert projection.shape == (5, 3)
    assert routes[:, 0].tolist() == [0] * 9 + [1] * 2
    assert doubled.shape == (13, 12)  # the padded value contributes nothing
    empty = torch.empty(0, dtype=torch.int32)
    zero = torch.empty((0, 0), dtype=torch.int32)
    for address in range(-1, 257):
        x = torch.tensor([2, 2 * address, -3] + [0] * 13, dtype=torch.int32)
        result, addresses = layer(
            x,
            memory,
            1,
            torch.arange(3),
            projection,
            routes,
            torch.arange(3, 16),
            w_out,
            empty,
            zero,
            empty,
            zero,
            halves,
            zero,
            zero,
            read=read,
        )
        valid = 0 <= address < 256
        byte = address if valid else 0
        expected = [byte, byte, *((byte >> b) & 1 for b in range(7)), 0, byte, byte]
        assert result[3:].tolist() == [2 * v for v in expected] + [-2]
        assert addresses.tolist() == [address if valid else -1] * 2


def test_shared_value_columns_merge_before_half_rounding_and_ignore_padding():
    query = torch.eye(2, dtype=torch.int32)
    projection = torch.cat([query, query])
    kinds = torch.tensor([[-1, -3], [-1, -2]])
    weights = torch.tensor([[3, 999, -5, 0], [0, -999, 1, -3]], dtype=torch.int32)
    shared, combined, routes = share_queries(projection, weights, kinds)
    assert torch.equal(shared, query)
    assert combined.tolist() == [[-2, 0], [1, -3]]
    assert routes.tolist() == [[0, -1], [0, -2]]
    with pytest.raises(ValueError, match="exceeds int32"):
        share_queries(projection, torch.full((1, 4), 2**30, dtype=torch.int32), kinds)


@pytest.mark.slow
def test_fixed_bank_matches_every_transformer_expert_and_read_counts():
    """Two native calling threads use distinct memory and stack scratch.

    Check all ABI readouts against eager attention/ReGLU layers, including
    half rounding and distinct reads of the same address in different layers.
    """
    machine = Cpu6502TransformerMachine()
    experts, _ = machine.core_experts()
    bank = bank_module(False)
    assert bank.n_experts == len(experts)
    prepared = []
    for expert in experts:
        layers = []
        for li in range(len(expert["model"].attn)):
            arrays = layer_arrays(expert["model"], li)
            names = WEIGHTS + (HALVES if "h_out" in arrays else ())
            layers.append(tuple(torch.from_numpy(arrays[name]) for name in names))
        prepared.append(layers)

    def run(seed):
        rng = np.random.default_rng(seed)
        memory = rng.integers(0, 256, 65536, dtype=np.uint8)
        enabled = ctypes.c_bool(True)
        counters = np.zeros(65536, dtype=np.uint32)
        context = bank.Context(memory.ctypes.data, ctypes.addressof(enabled), counters.ctypes.data)
        forward = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        )(bank.forward_address())
        calls = 0
        for ident, (expert, layers) in enumerate(zip(experts, prepared, strict=True)):
            model, spec = expert["model"], expert["step_spec"]
            keep = set(spec.get("outputs", {}).values())
            if "sequence_final" in spec:
                keep.update(spec["sequence_final"]["model"].values())
                keep.update(
                    s for writes in spec["sequence_writes"] for write in writes
                    for s in write.values() if s is not None
                )
                keep.update(spec["sequence_cycles"]["model"])
                if spec["sequence_halt"] is not None:
                    keep.add(spec["sequence_halt"])
            for _ in range(8):
                x = torch.zeros(model.tok.weight.shape[1], dtype=torch.int32)
                x[model.one_slot] = 2
                accumulator = int(rng.integers(0, 256))
                for name, slot in spec["inputs"].items():
                    if name == "is_state":
                        value = 1
                    elif name.startswith("a_bit"):
                        value = (accumulator >> int(name[5:])) & 1
                    elif name == "a":
                        value = accumulator
                    else:
                        value = int(rng.integers(0, 2 if name in ("nmi", "irq", "halted") else 256))
                    x[slot] = 2 * value
                for slot, _mask in spec["p_flags"]:
                    x[slot] = 2 * int(rng.integers(0, 2))
                expected, reads = x.clone(), collections.Counter()
                for weights in layers:
                    expected, addresses = layer(expected, torch.from_numpy(memory), 1, *weights)
                    reads.update(set(int(a) for a in addresses.tolist() if a >= 0))
                counters.fill(0)
                inputs = x.numpy().copy()
                original = inputs.copy()
                slots = expert["output_slots"]
                actual = np.full(len(slots) + 1, 0x5A5A5A5A, dtype=np.int32)
                assert forward(context.address, ident, inputs.ctypes.data, actual.ctypes.data)
                calls += 1
                assert slots == tuple(sorted(keep))
                np.testing.assert_array_equal(actual[: len(slots)], expected[list(slots)].numpy())
                assert actual[-1] == 0x5A5A5A5A
                np.testing.assert_array_equal(inputs, original)
                assert {int(a): int(counters[a]) for a in np.flatnonzero(counters)} == reads
        assert context.counts == (calls, 0)
        # Declining an invalid id does not touch the residual.
        original = actual.copy()
        assert not forward(context.address, bank.n_models, inputs.ctypes.data, actual.ctypes.data)
        np.testing.assert_array_equal(actual, original)
        assert context.counts == (calls, 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))


def test_counter_transition_weights_cover_every_licensed_state_and_limit():
    """All nonzero byte counters and byte-sized budgets, against literal steps.

    The graph has no memory heads or table of outcomes. Batched eager FX
    evaluates its actual matrices; the native bank is checked separately.
    """
    from transformer_nes.cpu6502.backends.weights.experts import build_counter_recurrence_experts

    values, limits = np.meshgrid(np.arange(1, 256), np.arange(256), indexing="ij")
    for expert in build_counter_recurrence_experts():
        spec = expert["step_spec"]
        register = "x" if expert["reg"] == 1 else "y"
        expected, counts = [], []
        for value in range(1, 256):
            path = [value]
            while (following := (path[-1] + expert["delta"]) & 255) != 0:
                path.append(following)
            count = np.minimum(np.arange(256), len(path) - 1)
            expected.extend(path[int(n)] for n in count)
            counts.extend(count)
        expected, counts = np.asarray(expected), np.asarray(counts)
        forward = weight_module(expert)
        positions = {slot: index for index, slot in enumerate(forward.readout_slots)}
        assert len(positions) < forward.input_width
        x = torch.zeros((forward.input_width, values.size), dtype=torch.int32)
        x[spec["inputs"][register]] = torch.from_numpy(values.ravel().astype(np.int32)) * 2
        x[spec["inputs"]["loop_limit"]] = torch.from_numpy(limits.ravel().astype(np.int32)) * 2
        for cycles in (5, 6):
            x[spec["inputs"]["loop_cycles"]] = 2 * cycles
            result, addresses = forward(x, torch.zeros(65536, dtype=torch.uint8))
            assert addresses.numel() == 0
            result = result.numpy() >> 1
            for slot, wanted in (
                (spec["sequence_final"]["model"][f"new_{register}"], expected),
                (spec["sequence_final"]["model"]["p_n"], expected >> 7),
                (spec["sequence_final"]["model"]["p_z"], np.zeros_like(expected)),
                (spec["outputs"]["iterations"], counts),
                (spec["sequence_cycles"]["model"][0], counts * cycles),
            ):
                np.testing.assert_array_equal(result[positions[slot]], wanted)


def test_machine_keeps_hook_context_alive_without_a_separate_handle():
    machine = Cpu6502TransformerMachine()
    image = bytearray(65536)
    image[0x8000:0x8003] = bytes([0xA9, 0x42, 0xEA])
    machine.load_image(image, start_pc=0x8000)
    gc.collect()
    machine.run(1, trace=False)
    assert machine.get_state()["a"] == 0x42
    assert machine._torch_forwards.counts == (1, 0)


def test_loaded_bank_survives_pruned_files_when_constructing_another_cpu(tmp_path, monkeypatch):
    from transformer_nes.cpu6502.backends.weights import torch_forward

    first = Cpu6502TransformerMachine()
    hook = bank_module(False)
    monkeypatch.setattr(torch_forward, "CACHE_DIR", str(tmp_path))

    def unexpected_child(*args, **kwargs):
        raise AssertionError("a loaded extension must not require a new compiler child")

    monkeypatch.setattr(torch_forward.subprocess, "run", unexpected_child)
    second = Cpu6502TransformerMachine()
    assert first._torch_forwards.hook is second._torch_forwards.hook is hook
    image = bytearray(65536)
    image[0x8000:0x8002] = bytes((0xA9, 0x42))
    second.load_image(image, start_pc=0x8000)
    second.run(1, trace=False)
    assert second.get_state()["a"] == 0x42


def test_compiler_child_rejects_stale_source_identity():
    from transformer_nes.cpu6502.backends.weights import torch_forward

    result = subprocess.run(
        [sys.executable, "-m", torch_forward.__name__, "0", "obsolete-source-digest"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "compiler sources changed" in result.stderr
    assert "restart the process" in result.stderr


def test_torch_engine_requires_its_bank_instead_of_falling_back():
    machine = Cpu6502TransformerMachine()
    image = bytearray(65536)
    image[0x8000:0x8002] = bytes([0xA9, 0x42])
    machine.load_image(image, start_pc=0x8000)
    machine.engine.engine.set_forward_hook(0, 0)
    with pytest.raises(RuntimeError, match="Torch transformer bank is not attached"):
        machine.run(1, trace=False)


@pytest.mark.parametrize("union_only", [False, True])
def test_read_requests_pause_without_committing_then_consume_the_reply(union_only):
    from transformer_nes.cpu6502.backends.reference import Cpu6502, instruction_cycles
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
            machine.engine.engine.set_run_state(
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
    machine.engine.engine.set_run_state(0x8000, 0, 0, 1, 0xFD, 0x24, 0, 0, 0, 0, 0, 0)
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
    machine.engine.engine.set_run_state(0x8000, 0, 0, 0, 0xFD, 0x20, 0, 0, 0, 0, 0, 0)
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
