"""Compiler cache identity, concurrent ownership, and token circuits compiled
through the model backend (model/backend.py) exactly."""

import multiprocessing
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from transformer_nes import build
from transformer_nes.graph.core import InputDimension, construction, current_graph, persist, reglu
from transformer_nes.model.backend import compile_model
from transformer_nes.model.construct import TokenGraph, construct_chip, describe


def _configure(root):
    build.CACHE_DIR = str(root)


@contextmanager
def _worker(target, *args, expected_exitcode=0):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=target, args=(*args, child))
    process.start()
    child.close()
    try:
        yield parent, process
        process.join(10)
        assert process.exitcode == expected_exitcode
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        process.close()


def _receive(connection):
    assert connection.poll(60), "compiler process did not respond"
    return connection.recv()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CACHE_DIR", str(tmp_path))
    return tmp_path


def compile_tokens(tokens, directory, *, memory_size=1, dtype="int32"):
    """Token types compiled through the model backend as one chip, into
    `directory`: the chip's engine and its generated source."""
    model = compile_model(
        {"t": construct_chip(tokens, memory_size=memory_size, dtype=dtype)}, Path(directory)
    )
    return model.chip("t"), (model.library / "chip_0.cpp").read_text()


@construction
def doubling():
    value = InputDimension("value")
    output = persist(2 * value, name="doubled")
    graph = current_graph().capture({"VALUE": value.as_expr()}, {"READOUT": output})
    return {"double": TokenGraph(graph, ["value"], ["doubled"])}


def _publish_many(root, prefix, connection):
    _configure(root)
    connection.send("ready")
    assert connection.recv() == "publish"
    for index in range(40):
        build.manifest_put(f"{prefix}:{index}", {"digest": "1111111111111111"})


def test_manifest_preserves_concurrent_publishers(cache):
    with (
        _worker(_publish_many, cache, "first") as (first, _),
        _worker(_publish_many, cache, "second") as (second, _),
    ):
        assert _receive(first) == _receive(second) == "ready"
        first.send("publish")
        second.send("publish")
    assert set(build.manifest_read()) == {
        f"{prefix}:{index}" for prefix in ("first", "second") for index in range(40)
    }


def test_cpu_cache_hit_rechecks_after_waiting_for_artifact(cache, monkeypatch):
    from transformer_nes.cpu6502 import engine

    monkeypatch.setattr(build, "CACHE_DIR", str(cache))
    digest = "1111111111111111"
    header = cache / f"engine_{digest}.hpp"
    header.touch()
    build.manifest_put("cpu", {"digest": digest, "compile_shards": engine.compile_shards()})

    lock = build.artifact_lock

    @contextmanager
    def pruned_before_acquisition(locked_digest, *, blocking=True):
        # The reader already took its manifest snapshot. Replace it before
        # it can acquire the artifact, just as a competing publisher can.
        if locked_digest == digest and blocking:
            build.manifest_put("cpu", {"digest": "2222222222222222"})
            assert not header.exists()
        with lock(locked_digest, blocking=blocking) as acquired:
            yield acquired

    monkeypatch.setattr(build, "artifact_lock", pruned_before_acquisition)
    assert engine.GeneratedWeightsEngine.from_cache("cpu") is None


def _compile_doubling(root, connection):
    _configure(root)
    engine, _source = compile_tokens(doubling(), Path(root) / "models")
    connection.send(int(engine.step([21])[0]))


def test_concurrent_identical_builds_publish_one_library(cache):
    with (
        _worker(_compile_doubling, cache) as (first, _),
        _worker(_compile_doubling, cache) as (second, _),
    ):
        assert [_receive(first), _receive(second)] == [42, 42]
    published = sorted(p.name for p in (cache / "models").iterdir())
    assert len(published) == 1 and published[0].startswith("nes_")


def _abandoned_build(root, digest, connection):
    _configure(root)
    with build.artifact_lock(digest):
        directory = Path(root) / f"build_{digest}"
        directory.mkdir()
        (directory / "lock").touch()  # Torch's file baton is not an OS lock.
        (Path(root) / f"graph_{digest}.cpp").write_text("unfinished source")
        connection.send("locked")
        connection.recv()  # The test kills this process while it owns the lock.


def _recover_build(root, digest, connection):
    _configure(root)
    with build.artifact_lock(digest):
        source = Path(root) / f"graph_{digest}.cpp"
        source.write_text(
            "#include <pybind11/pybind11.h>\n"
            'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("answer", [] { return 42; }); }\n'
        )
        module = build.load_module(f"cache_recovery_{digest}", digest, [str(source)])
        assert module.answer() == 42
        connection.send("recovered")


def test_killed_builder_releases_ownership_and_stale_torch_baton(cache):
    digest = "3333333333333333"
    with _worker(_abandoned_build, cache, digest, expected_exitcode=-signal.SIGTERM) as (
        connection,
        process,
    ):
        assert _receive(connection) == "locked"
        # Pruning must skip this digest while the other process owns it.
        build.manifest_put("pruner", {"digest": "ffffffffffffffff"})
        assert (cache / f"build_{digest}" / "lock").exists()
        process.terminate()
        process.join(10)

    with _worker(_recover_build, cache, digest) as (connection, _):
        assert _receive(connection) == "recovered"
    # No manifest entry was published for the abandoned/recovered artifact.
    # Once its owner exits, ordinary pruning can reclaim the whole group.
    build.manifest_put("pruner", {"digest": "ffffffffffffffff"})
    assert not (cache / f"graph_{digest}.cpp").exists()
    assert not (cache / f"build_{digest}").exists()


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "PROJECT_DIR", tmp_path)
    root = tmp_path / "transformer_nes"
    root.mkdir()
    files = {
        "__init__.py": "from . import initialization\n",
        "initialization.py": "value = 1\n",
        "builder.py": '''
"""from transformer_nes import documentation_only"""
from . import helper as h
if True:
    from transformer_nes.conditional import enabled
try:
    import transformer_nes.first as first, transformer_nes.second as second
except ImportError:
    pass

def optional():
    from . import lazy

async def also_optional():
    from . import lazy
''',
        **{
            f"{name}.py": "enabled = True\n"
            for name in (
                "helper",
                "conditional",
                "first",
                "second",
                "lazy",
                "documentation_only",
                "unrelated",
            )
        },
        "runtime.inc": "int value = 1;\n",
    }
    for name, text in files.items():
        (root / name).write_text(text)
    build.source_digest.cache_clear()
    yield root
    build.source_digest.cache_clear()


@pytest.mark.parametrize(
    "name,changed",
    [
        (name, True)
        for name in (
            "helper.py",
            "conditional.py",
            "initialization.py",
            "first.py",
            "second.py",
            "runtime.inc",
        )
    ]
    + [(name, False) for name in ("lazy.py", "documentation_only.py", "unrelated.py")],
)
def test_cache_identity_follows_source_dependencies(package, name, changed):
    def digest():
        build.source_digest.cache_clear()
        return build.source_digest(
            "transformer_nes.builder", files=(str(package / "runtime.inc"),)
        )

    before = digest()
    with (package / name).open("a") as source:
        source.write("\n# changed\n")
    assert (digest() != before) == changed


def test_generic_packages_do_not_import_a_cpu():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import pkgutil
import sys

class NoCpu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('transformer_nes.cpu6502'):
            raise AssertionError(f'generic compiler imported {fullname}')

sys.meta_path.insert(0, NoCpu())
for name in ('graph', 'model'):
    package = importlib.import_module(f'transformer_nes.{name}')
    for module in pkgutil.iter_modules(package.__path__):
        if not module.ispkg:
            importlib.import_module(f'{package.__name__}.{module.name}')
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_library_identity_covers_memory_size_and_outlives_replacement(cache):
    small, _ = compile_tokens(doubling(), cache / "models", memory_size=4096)
    assert list(small.step([21])) == [42]
    small.write(4095, 7)
    with pytest.raises(IndexError):
        small.write(4096, 7)
    (library,) = (cache / "models").iterdir()
    built = library.stat().st_mtime_ns

    cached, _ = compile_tokens(doubling(), cache / "models", memory_size=4096)
    assert [p.stat().st_mtime_ns for p in (cache / "models").iterdir()] == [built]
    assert list(cached.step([13])) == [26]

    large, _ = compile_tokens(doubling(), cache / "models", memory_size=8192)
    (replacement,) = (cache / "models").iterdir()
    assert replacement != library
    large.write(8191, 9)
    assert list(large.step([17])) == [34]
    # An engine of the replaced library stays usable: its mapping outlives the files.
    assert list(small.step([5])) == [10]


def test_token_projection_removes_unused_computation():
    @construction
    def token():
        value = InputDimension("value")
        kept = persist(2 * value, name="kept")
        discarded = persist(reglu(value, value + 1), name="discarded")
        graph = current_graph().capture({"VALUE": value.as_expr()}, {"READOUT": kept + discarded})
        return TokenGraph(graph, ("value",), ("kept", "discarded"))

    full = token()
    projected = full.project(("kept",))
    assert projected.inputs == full.inputs
    assert projected.outputs == ("kept",)
    assert len(projected.program_graph.all_dims) < len(full.program_graph.all_dims)
    assert "discarded" not in {d.name for d in projected.program_graph.all_dims}
    with pytest.raises(ValueError, match="existing token outputs"):
        full.project(("missing",))


def test_float32_weights_preserve_real_inputs_trace_and_ablation(cache):
    import ctypes

    import numpy as np

    from transformer_nes.graph.fold import GraphFolder
    from transformer_nes.model.construct import construct

    @construction
    def graphs():
        x = InputDimension("x")
        y = persist(0.125 * reglu(x + 0.25, x - 0.375) - 0.9995 * x, name="y")
        graph = current_graph().capture({"VALUE": x.as_expr()}, {"READOUT": y})
        return {"real": TokenGraph(graph, ("x",), ("y",))}

    engine, _ = compile_tokens(graphs(), cache / "models", dtype="float32")
    assert engine.dtype == np.dtype("float32") and engine.chip.f_bits == 0
    for x in np.linspace(-2, 2, 65):
        expected = 0.125 * (x + 0.25) * max(x - 0.375, 0) - 0.9995 * x
        np.testing.assert_allclose(engine.step([x]), [expected], rtol=0, atol=3e-7)
    abi = engine.abi()
    trace = ctypes.PYFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(abi[3])
    inputs = np.array([0.8125], dtype=np.float32)
    residuals = np.zeros((abi[4], abi[5]), dtype=np.float32)
    trace(abi[0], inputs.ctypes.data, residuals.ctypes.data)
    token = graphs()["real"]
    fold = GraphFolder(token.program_graph, root_names=["y"], structural_reads=True).fold({})
    info = construct(fold.program_graph)[2]
    slot = next(slot for dim, slot in info.slot_of.items() if dim.name == "y")
    np.testing.assert_allclose(residuals[-1, slot], engine.step(inputs)[0], rtol=0, atol=3e-7)
    engine.engine.ablate(slot, info.n_layers, 0.123456789)
    assert engine.step(inputs).tolist() == [float(np.float32(0.123456789))]
    engine.engine.ablate()
    np.testing.assert_array_equal(engine.step(inputs), residuals[-1, slot:slot + 1])


@pytest.mark.parametrize(("gate", "input_value"), [(1, 2), (0.5, 4)])
def test_constant_one_survives_implicit_linear_gates(gate, input_value):
    import torch

    from transformer_nes.model.compiler import weight_module
    from transformer_nes.model.construct import construct

    @construction
    def token():
        value = InputDimension("value")
        result = reglu(value, gate)
        for _ in range(4):
            result = reglu(result, result)
        result = persist(result, name="result")
        graph = current_graph().capture({"VALUE": value.as_expr()}, {"READOUT": result})
        return TokenGraph(graph, ("value",), ("result",))

    token = token()
    pg = token.program_graph
    _model, _idx, info = construct(pg)
    assert (pg.one, info.slot_of[pg.one], 0, info.n_layers + 1) in info.lives
    compiled = describe(token, pg, pg.all_dims)
    forward = weight_module(compiled)
    assert forward.f_bits == int(gate == 0.5)
    result, _reads = forward(
        torch.tensor([input_value], dtype=torch.int32), torch.zeros(65536, dtype=torch.uint8)
    )
    assert result.tolist() == [(1 << forward.f_bits) * 2**16]


@pytest.mark.parametrize("reassociate_residuals", [False, True])
def test_integer_expansion_preserves_memory_bounds_and_half_rounding(reassociate_residuals, cache):
    import numpy as np
    import torch

    from transformer_nes.graph.fold import GraphFolder
    from transformer_nes.ppu2c02.space import Memory

    @construction
    def graphs():
        address = InputDimension("address")
        offset = InputDimension("offset")
        write, mem_address, mem_value = (InputDimension(n) for n in ("write", "mem_address", "mem_value"))
        memory = Memory(write, mem_address, mem_value)
        byte = memory.read(address, (0,), "byte")
        low = memory.bits(byte, "byte")[7]
        result = persist(0.5 * (byte - low) + reglu(offset, 1), name="result")
        graph = current_graph().capture(
            {"VALUE": address + offset, "WRITE": write + mem_address + mem_value}, {"READOUT": result}
        )
        return {"read": TokenGraph(
            graph, ("address", "offset"), ("result",),
            reassociate_residuals=reassociate_residuals,
        )}

    engine, _ = compile_tokens(graphs(), cache / "models", memory_size=256)
    for address in range(256):
        engine.write(address, address)
    # Exercise both the FX and native paths, including negative half products,
    # invalid addresses and the fixed-point rescaling at the raw ABI.
    token = graphs()["read"]
    folded = GraphFolder(token.program_graph, root_names=list(token.outputs), structural_reads=True).fold({})
    expert = describe(token, folded.program_graph, list(folded.dim_of_name.values()))
    from transformer_nes.model.compiler import weight_module

    forward = weight_module(expert, memory_size=256)
    memory = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    for address in (-1, *range(256), 256):
        for offset in (-3, 0, 5):
            expected = [((address >> 1) if 0 <= address < 256 else 0) + offset]
            np.testing.assert_array_equal(engine.step([address, offset]), expected)
            output, _ = forward(torch.tensor([address, offset], dtype=torch.int32), memory)
            assert output.tolist() == [v << forward.f_bits for v in expected]


def test_integer_residual_simplification_preserves_int32_overflow(cache):
    @construction
    def graphs():
        value = InputDimension("value")
        result = persist(reglu(value, value) + 17 * value, name="result")
        graph = current_graph().capture({"VALUE": value.as_expr()}, {"READOUT": result})
        return {"overflow": TokenGraph(graph, ("value",), ("result",))}

    engine, _ = compile_tokens(graphs(), cache / "models")
    for value in (-2**31, -65536, -1, 0, 1, 46341, 65536, 2**31 - 1):
        expected = (value * max(0, value) + 17 * value + 2**31) % 2**32 - 2**31
        assert engine.step([value]).tolist() == [expected]


def _cached_body():
    return "void circuit_0() {}", ["#include <cstdint>"], dict(
        width=1, reads=0, arguments=1, checks_division=False,
    )


def _identity_extractor(source, index, *, scalar="int32_t"):
    return source, scalar, index + 1


def test_kernel_identity_tracks_code_not_source_locations():
    import types

    from transformer_nes.build import kernel_fingerprint

    def key(extractor, **overrides):
        return kernel_fingerprint(
            ("graph",), overrides.pop("config", {"cpp.threads": 1}),
            **(dict(memory_size=256, scalar="int32_t", extractor=extractor) | overrides),
        )

    moved = types.FunctionType(
        _identity_extractor.__code__.replace(co_filename="moved.py", co_firstlineno=900), globals(),
    )
    moved.__kwdefaults__ = _identity_extractor.__kwdefaults__
    changed = types.FunctionType(
        _identity_extractor.__code__.replace(
            co_consts=tuple(2 if type(c) is int and c == 1 else c for c in _identity_extractor.__code__.co_consts)
        ), globals(),
    )
    changed.__kwdefaults__ = _identity_extractor.__kwdefaults__
    assert key(moved) == key(_identity_extractor)
    assert key(changed) != key(_identity_extractor)
    assert key(moved, memory_size=512) != key(moved)
    assert key(moved, scalar="float") != key(moved)
    assert key(moved, config={"cpp.threads": 2}) != key(moved)


def test_kernel_cache_recovers_corrupt_entries_and_evicts_old_sources(cache, monkeypatch):
    from transformer_nes import build

    lowered = []

    def lower():
        lowered.append(1)
        return _cached_body()

    first, second = "a" * 64, "b" * 64
    assert build.cached_kernel(first, lower)[-1] is False
    assert build.cached_kernel(first, lower)[-1] is True
    path = cache / "kernels" / f"{first}.json"
    size = path.stat().st_size
    path.write_text('{"body": "truncated')
    assert build.cached_kernel(first, lower)[-1] is False
    assert len(lowered) == 2
    monkeypatch.setattr(build, "MAX_KERNEL_BYTES", size + 1)
    assert build.cached_kernel(second, lower)[-1] is False
    assert not path.exists()
    assert build.cached_kernel(second, lower)[-1] is True


def _shared_kernel(root, may_compile, connection):
    from transformer_nes import build

    _configure(root)
    connection.send("started")

    def lower():
        assert may_compile, "the second process must reuse the completed kernel"
        connection.send("lowering")
        assert connection.recv() == "finish"
        return _cached_body()

    body, headers, meta, hit = build.cached_kernel("c" * 64, lower)
    connection.send((body, headers, meta, hit))


def test_kernel_cache_owns_compilation_across_processes(cache):
    with _worker(_shared_kernel, cache, True) as (first, _):
        assert _receive(first) == "started"
        assert _receive(first) == "lowering"
        with _worker(_shared_kernel, cache, False) as (second, _):
            assert _receive(second) == "started"
            assert not second.poll(0.2)
            first.send("finish")
            a, b = _receive(first), _receive(second)
            assert a[:-1] == b[:-1] == _cached_body()
            assert a[-1] is False and b[-1] is True


def test_cached_circuits_relocate_between_banks_without_lowering(cache, monkeypatch):
    from transformer_nes import build as compiler

    monkeypatch.setattr(build, "CACHE_DIR", str(cache))

    @construction
    def graphs(reverse=False, changed=False):
        value = InputDimension("value")
        doubled = persist(2 * value, name="doubled")
        square = persist(reglu(value, value) + (value if changed else 0), name="square")
        graph = current_graph().capture(
            {"VALUE": value.as_expr()}, {"READOUT": doubled + square}
        )
        tokens = {
            "double": TokenGraph(graph, ("value",), ("doubled",)),
            "square": TokenGraph(graph, ("value",), ("square",)),
        }
        return dict(reversed(list(tokens.items()))) if reverse else tokens

    original, _ = compile_tokens(graphs(), cache / "original")
    assert original.step([7], original.model_id("double")).tolist() == [14]
    assert original.step([7], original.model_id("square")).tolist() == [49]

    real_compile = compiler.cached_kernel

    def unexpected_lowering(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        assert result[-1], "identical SSA circuits must survive a new bank identity"
        return result

    monkeypatch.setattr(compiler, "cached_kernel", unexpected_lowering)
    # Different token order and model ids, library identity and unused RAM
    # extent: bind each cached body's circuit name to its new index.
    reordered, _ = compile_tokens(graphs(True), cache / "reordered", memory_size=4096)
    for value in (-7, 0, 11):
        assert reordered.step([value], reordered.model_id("double")).tolist() == [2 * value]
        assert reordered.step([value], reordered.model_id("square")).tolist() == [value * max(0, value)]

    calls = []
    def count_lowering(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        if not result[-1]:
            calls.append(1)
        return result

    monkeypatch.setattr(compiler, "cached_kernel", count_lowering)
    changed, _ = compile_tokens(graphs(changed=True), cache / "changed")
    assert len(calls) == 1
    assert changed.step([7], changed.model_id("double")).tolist() == [14]
    assert changed.step([7], changed.model_id("square")).tolist() == [56]


def test_fixed_read_cache_shares_valid_extents_and_keeps_out_of_bounds_zero(cache, monkeypatch):
    from transformer_nes import build as compiler
    from transformer_nes.ppu2c02.space import Memory

    monkeypatch.setattr(build, "CACHE_DIR", str(cache))

    @construction
    def graphs():
        offset = InputDimension("offset")
        write, address, value = (InputDimension(n) for n in ("write", "address", "value"))
        byte = Memory(write, address, value).read(7, (0,), "byte")
        result = persist(byte + 2 * offset, name="result")
        graph = current_graph().capture(
            {"VALUE": offset.as_expr(), "WRITE": write + address + value}, {"READOUT": result}
        )
        return {"read": TokenGraph(graph, ("offset",), ("result",))}

    real_compile, calls = compiler.cached_kernel, []

    def count_lowering(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        if not result[-1]:
            calls.append(1)
        return result

    monkeypatch.setattr(compiler, "cached_kernel", count_lowering)
    large, _ = compile_tokens(graphs(), cache / "large", memory_size=256)
    large.write(7, 215)
    assert large.step([3]).tolist() == [221]
    assert len(calls) == 1
    small, _ = compile_tokens(graphs(), cache / "small", memory_size=8)
    small.write(7, 13)
    assert small.step([3]).tolist() == [19]
    assert len(calls) == 1
    outside, _ = compile_tokens(graphs(), cache / "outside", memory_size=4)
    assert outside.step([3]).tolist() == [6]
    assert len(calls) == 2


def test_shared_neurons_emit_once_without_importing_inductor(cache, monkeypatch):
    import builtins

    monkeypatch.setattr(build, "CACHE_DIR", str(cache))
    original_import = builtins.__import__

    def no_inductor(name, *args, **kwargs):
        if name.startswith(("torch._inductor", "torch._dynamo")):
            raise AssertionError(f"scalar lowering imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_inductor)

    @construction
    def graphs():
        x = InputDimension("x")
        shared = reglu(x, x)
        outputs = [persist(shared + (i + 1) * x, name=f"out{i}") for i in range(64)]
        graph = current_graph().capture({"VALUE": x.as_expr()}, {"READOUT": sum(outputs)})
        return {"fanout": TokenGraph(graph, ("x",), tuple(f"out{i}" for i in range(64)))}

    engine, source = compile_tokens(graphs(), cache / "models")
    assert source.count(" > 0 ?") == 1
    assert "scratch_" not in source and "phase_" not in source
    for value in (-65536, -1, 0, 1, 46341, 65536):
        expected = [
            (value * max(0, value) + (i + 1) * value + 2**31) % 2**32 - 2**31
            for i in range(64)
        ]
        assert engine.step([value]).tolist() == expected


def test_ppu_source_stays_compact():
    from transformer_nes.model.backend import chip_source
    from transformer_nes.nes.model import NES

    source = chip_source(0, NES.load().chips["ppu"])
    # Real rendering, fetch, status and register experts previously emitted
    # over 20 MiB, then 4.1 MiB with scalar SSA, then about 2.7 MiB for one
    # bank serving every cart. With specializations that do not pay gone and
    # shared sums computed once it is about 0.75 MiB: preserve the
    # shared-fragment and compact scalar representation.
    assert len(source.encode()) < 1024**2


def test_float32_ssa_matches_fx_bits_at_rounding_boundaries(cache):
    import numpy as np
    import torch

    from transformer_nes.graph.fold import GraphFolder
    from transformer_nes.model.compiler import weight_module

    @construction
    def graphs():
        x = InputDimension("x")
        y = persist(reglu(1.0000001192092896 * x + 2**-24, x + 2) - 0.9995 * x, name="y")
        pg = current_graph().capture({"VALUE": x.as_expr()}, {"READOUT": y})
        return {"real": TokenGraph(pg, ("x",), ("y",))}

    engine, _ = compile_tokens(graphs(), cache / "models", dtype="float32")
    token = graphs()["real"]
    folded = GraphFolder(token.program_graph, root_names=list(token.outputs), structural_reads=True).fold({})
    model = describe(token, folded.program_graph, list(folded.dim_of_name.values()), dtype="float32")
    forward = weight_module(model, memory_size=1)
    values = np.concatenate([
        np.array([-0.0, 0.0, -2, 2**-24, 1, np.nextafter(np.float32(1), np.float32(2))], dtype=np.float32),
        np.random.default_rng(819).uniform(-100, 100, 256).astype(np.float32),
    ])
    expected, _ = forward(torch.from_numpy(values.reshape(1, -1)), torch.zeros(1, dtype=torch.uint8))
    actual = np.array([engine.step([x])[0] for x in values], dtype=np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.numpy().ravel().view(np.uint32))


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_shared_fragments_preserve_bound_values_and_projection_grouping(dtype, cache):
    import numpy as np
    import torch

    @construction
    def graphs():
        x, y = InputDimension("x"), InputDimension("y")
        value = x
        for _ in range(24):
            value = persist(reglu(value + 2, y + 1))
        result = persist(value, name="result")
        graph = current_graph().capture({"VALUE": x + y}, {"READOUT": result})
        return {
            "xy": TokenGraph(graph, ("x", "y"), ("result",), reassociate_residuals=False),
            "yx": TokenGraph(graph, ("y", "x"), ("result",), reassociate_residuals=False),
        }

    engine, source = compile_tokens(graphs(), cache / "models", dtype=dtype)
    assert "always_inline" in source
    pairs = [(-0.0, -1), (0, 0), (1, 1), (-7, 1), (46341, 2), (2**31 - 1, 0)]
    if dtype == "int32":
        def wrap(value):
            return (int(value) + 2**31) % 2**32 - 2**31

        expected = []
        for x, y in pairs:
            for _ in range(24):
                x = wrap(wrap(x + 2) * wrap(y + 1)) if wrap(y + 1) > 0 else 0
            expected.append(x)
    else:
        from transformer_nes.graph.fold import GraphFolder
        from transformer_nes.model.compiler import weight_module

        token = graphs()["xy"]
        folded = GraphFolder(token.program_graph, root_names=list(token.outputs), structural_reads=True).fold({})
        model = describe(token, folded.program_graph, list(folded.dim_of_name.values()), dtype=dtype)
        forward = weight_module(model, memory_size=1)
        result, _ = forward(torch.tensor(np.array(pairs).T, dtype=torch.float32), torch.zeros(1, dtype=torch.uint8))
        expected = result.numpy().ravel()
    for name in ("xy", "yx"):
        actual = np.array([engine.step(pair if name == "xy" else pair[::-1], engine.model_id(name))[0]
                           for pair in pairs], dtype=np.dtype(dtype))
        if dtype == "float32":
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        else:
            np.testing.assert_array_equal(actual, expected)
