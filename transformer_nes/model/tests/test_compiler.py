"""Compiler cache identity, concurrent ownership and reusable graph execution."""

import multiprocessing
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from transformer_nes.graph.core import InputDimension, construction, current_graph, persist, reglu
from transformer_nes.model import build, graph_engine, sources
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph


def _configure(root):
    build.CACHE_DIR = str(root)
    build.MANIFEST_PATH = str(Path(root) / "manifest.json")


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
    monkeypatch.setattr(build, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    return tmp_path


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
    from transformer_nes.cpu6502.backends.weights import engine

    monkeypatch.setattr(engine, "CACHE_DIR", str(cache))
    digest = "1111111111111111"
    header = cache / f"engine_{digest}.hpp"
    header.touch()
    build.manifest_put("cpu", {"digest": digest, "compile_shards": engine.native.compile_shards()})

    @contextmanager
    def pruned_before_acquisition(locked_digest):
        assert locked_digest == digest
        # The reader already took its manifest snapshot. Replace it before
        # it can acquire the artifact, just as a competing publisher can.
        build.manifest_put("cpu", {"digest": "2222222222222222"})
        assert not header.exists()
        with build.artifact_lock(digest):
            yield

    monkeypatch.setattr(engine, "artifact_lock", pruned_before_acquisition)
    assert engine.GeneratedWeightsEngine.from_cache("cpu") is None


def _compile_graph(root, size, connection):
    import torch.utils.cpp_extension as extension

    from transformer_nes.graph.core import InputDimension, construction, current_graph, persist
    from transformer_nes.model import graph_engine
    from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph

    _configure(root)
    graph_engine.CACHE_DIR = str(root)
    original_load = extension.load

    def paused_load(*args, **kwargs):
        connection.send(("building", Path(kwargs["build_directory"]).name.removeprefix("build_")))
        assert connection.recv() == "compile"
        return original_load(*args, **kwargs)

    extension.load = paused_load

    @construction
    def graphs():
        value = InputDimension("value")
        output = persist(2 * value, name="doubled")
        graph = current_graph().capture({"VALUE": value.as_expr()}, {"READOUT": output})
        return {"double": TokenGraph(graph, ["value"], ["doubled"])}

    tag = f"parallel:{size}"
    engine = CompiledGraph(graphs, tag=tag, sources=(), memory_size=size)
    assert list(engine.step([21])) == [42]

    def unexpected_rebuild():
        raise AssertionError("a completed graph should be reusable without construction")

    cached = CompiledGraph(unexpected_rebuild, tag=tag, sources=(), memory_size=size)
    assert list(cached.step([13])) == [26]
    connection.send(engine.digest)


def test_parallel_builds_survive_pruning_and_publish_only_after_loading(cache):
    obsolete = cache / "graph_0000000000000000.cpp"
    obsolete.write_text("obsolete")
    with (
        _worker(_compile_graph, cache, 4096) as (first, _),
        _worker(_compile_graph, cache, 8192) as (second, _),
    ):
        # Both must enter compilation before either finishes: independent
        # digests cannot be serialized by the manifest lock.
        stages = [_receive(first), _receive(second)]
        assert all(stage == "building" for stage, _ in stages)
        digests = [digest for _, digest in stages]
        assert len(set(digests)) == 2
        assert build.manifest_read() == {}
        build.manifest_put("pruner", {"digest": "ffffffffffffffff"})
        assert not obsolete.exists()
        for digest in digests:
            assert (cache / f"graph_{digest}.cpp").exists()
            assert (cache / f"build_{digest}").is_dir()
        first.send("compile")
        second.send("compile")
        assert [_receive(first), _receive(second)] == digests
    manifest = build.manifest_read()
    assert manifest["graph:parallel:4096"]["digest"] == digests[0]
    assert manifest["graph:parallel:8192"]["digest"] == digests[1]
    assert "pruner" in manifest


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
    monkeypatch.setattr(sources, "ROOT", str(tmp_path))
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
    sources.source_digest.cache_clear()
    yield root
    sources.source_digest.cache_clear()


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
        sources.source_digest.cache_clear()
        return sources.source_digest(
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


def test_graph_cache_respects_memory_size(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(build, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(graph_engine, "CACHE_DIR", str(tmp_path))
    builds = 0

    @construction
    def graphs():
        nonlocal builds
        builds += 1
        value = InputDimension("value")
        output = persist(2 * value, name="doubled")
        graph = current_graph().capture({"VALUE": value.as_expr()}, {"READOUT": output})
        return {"double": TokenGraph(graph, ["value"], ["doubled"])}

    def compile_engine(size):
        return CompiledGraph(graphs, tag="cache_test", sources=(), memory_size=size)

    small = compile_engine(4096)
    assert list(small.step([21])) == [42]
    small.write(4095, 7)
    with pytest.raises(IndexError):
        small.write(4096, 7)

    cached = compile_engine(4096)
    assert builds == 1
    assert cached.digest == small.digest
    assert list(cached.step([13])) == [26]

    def failed_load(*args, **kwargs):
        raise RuntimeError("compiler failed")

    with monkeypatch.context() as failed_build:
        failed_build.setattr(graph_engine, "load_module", failed_load)
        with pytest.raises(RuntimeError, match="compiler failed"):
            compile_engine(8192)
    assert build.manifest_read()["graph:cache_test"]["digest"] == small.digest
    assert compile_engine(4096).digest == small.digest
    assert builds == 2

    large = compile_engine(8192)
    assert builds == 3
    assert large.digest != small.digest
    large.write(8191, 9)
    assert list(large.step([17])) == [34]

    # The original instance remains usable after replacing its cache entry.
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


def test_float32_weights_preserve_real_inputs_trace_and_ablation():
    import ctypes

    import numpy as np

    @construction
    def graphs():
        x = InputDimension("x")
        y = persist(0.125 * reglu(x + 0.25, x - 0.375) - 0.9995 * x, name="y")
        graph = current_graph().capture({"VALUE": x.as_expr()}, {"READOUT": y})
        return {"real": TokenGraph(graph, ("x",), ("y",))}

    engine = CompiledGraph(graphs, tag="real_test", sources=(), memory_size=1, dtype="float32")
    assert engine.dtype == np.dtype("float32") and engine.f_bits == 0
    for x in np.linspace(-2, 2, 65):
        expected = 0.125 * (x + 0.25) * max(x - 0.375, 0) - 0.9995 * x
        np.testing.assert_allclose(engine.step([x]), [expected], rtol=0, atol=3e-7)
    abi = engine.abi()
    trace = ctypes.PYFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(abi[3])
    inputs = np.array([0.8125], dtype=np.float32)
    residuals = np.zeros((abi[4], abi[5]), dtype=np.float32)
    trace(abi[0], inputs.ctypes.data, residuals.ctypes.data)
    slot = next(slot for dim, slot in engine.slot_of.items() if dim.name == "y")
    np.testing.assert_allclose(residuals[-1, slot], engine.step(inputs)[0], rtol=0, atol=3e-7)
    engine.engine.ablate(slot, engine.n_layers, 0.123456789)
    assert engine.step(inputs).tolist() == [float(np.float32(0.123456789))]
    engine.engine.ablate()
    np.testing.assert_array_equal(engine.step(inputs), residuals[-1, slot:slot + 1])


@pytest.mark.parametrize(("gate", "input_value"), [(1, 2), (0.5, 4)])
def test_constant_one_survives_implicit_linear_gates(gate, input_value):
    import torch

    from transformer_nes.model.graph_engine import construct
    from transformer_nes.model.torch_codegen import weight_module

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
    compiled = CompiledGraph._compile(token, pg, pg.all_dims)
    forward = weight_module(compiled)
    assert forward.f_bits == int(gate == 0.5)
    result, _reads = forward(
        torch.tensor([input_value], dtype=torch.int32), torch.zeros(65536, dtype=torch.uint8)
    )
    assert result.tolist() == [(1 << forward.f_bits) * 2**16]
