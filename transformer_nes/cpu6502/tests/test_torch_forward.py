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
        forward = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)(
            bank.forward_address()
        )
        calls = 0
        for ident, (expert, layers) in enumerate(zip(experts, prepared, strict=True)):
            model, spec = expert["model"], expert["step_spec"]
            keep = set(spec["sequence_final"]["model"].values())
            keep.update(s for writes in spec["sequence_writes"] for pair in writes for s in pair)
            keep.update(spec["sequence_cycles"]["model"])
            keep.update(
                s
                for s in (
                    spec["sequence_halt"],
                    spec["sequence_control"]["predicate"],
                    spec["sequence_control"]["target_dimension"],
                )
                if s is not None
            )
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
                actual = x.numpy().copy()
                assert forward(context.address, ident, actual.ctypes.data)
                calls += 1
                for slot in keep:
                    assert actual[slot] == int(expected[slot]), (seed, ident, slot)
                assert {int(a): int(counters[a]) for a in np.flatnonzero(counters)} == reads
        assert context.counts == (calls, 0)
        # Declining an invalid id does not touch the residual.
        original = actual.copy()
        assert not forward(context.address, bank.n_models, actual.ctypes.data)
        np.testing.assert_array_equal(actual, original)
        assert context.counts == (calls, 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))


def test_machine_keeps_hook_context_alive_without_a_separate_handle():
    machine = Cpu6502TransformerMachine()
    image = bytearray(65536)
    image[0x8000:0x8003] = bytes([0xA9, 0x42, 0xEA])
    machine.load_image(image, start_pc=0x8000)
    gc.collect()
    machine.run(1, trace=False)
    assert machine.get_state()["a"] == 0x42
    assert machine._torch_forwards.counts == (1, 0)


def test_torch_engine_requires_its_bank_instead_of_falling_back():
    machine = Cpu6502TransformerMachine()
    image = bytearray(65536)
    image[0x8000:0x8002] = bytes([0xA9, 0x42])
    machine.load_image(image, start_pc=0x8000)
    machine.engine.engine.set_forward_hook(0, 0)
    with pytest.raises(RuntimeError, match="Torch transformer bank is not attached"):
        machine.run(1, trace=False)
