"""Compile token graphs and folded experts into a shared-memory native bank.

Model 0 is the primary graph; callers select experts whose pinned inputs hold.
Named token inputs/outputs define the raw ABI: int32 pointers for discrete
banks, float pointers for memory-free float32 banks. Trace and ablation operate
on the primary graph's residual stream.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from transformer_nes import build
from transformer_nes.graph.fold import GraphFolder
from transformer_nes.model.compiler import bank_source
from transformer_nes.model.construct import TokenGraph, construct, describe

logger = logging.getLogger(__name__)
NATIVE_DIR = Path(__file__).with_name("native")


def generate_graph_source(models: list[dict], memory_size: int = 0x10000) -> tuple[str, int]:
    f_bits = max(m["fraction_bits"] for m in models)
    for model in models:
        model["fraction_bits"] = f_bits
    table = ["struct Token { int n_in, n_out; };", "static const Token tokens[] = {"]
    table.extend(f"{{{len(m['input_slots'])}, {len(m['output_slots'])}}}," for m in models)
    table.append("};")
    adapter = (NATIVE_DIR / "graph.cpp").read_text()
    decode = """
        for (int i = 0; i < tokens[model].n_out; ++i) {
            if (result[i] & ((1 << @F_BITS@) - 1)) throw std::runtime_error("fractional readout value");
            out[i] = result[i] >> @F_BITS@;
        }
    """ if f_bits else ""
    for key, value in {
        "@DECODE@": decode,
        "@TOKEN_TABLE@": "\n".join(table),
        "@MEMORY@": str(memory_size),
        "@F_BITS@": str(f_bits),
        "@MAX_INPUT@": str(max(len(m["input_slots"]) for m in models) + 3),
        "@PRIMARY_WIDTH@": str(models[0]["d_model"]),
        "@MAX_OUTPUT@": str(max(len(m["output_slots"]) for m in models)),
        "@TRACE_ROWS@": str(models[0]["n_layers"] + 1),
    }.items():
        adapter = adapter.replace(key, value)
    return bank_source(models, adapter, memory_size=memory_size), f_bits


@dataclass(frozen=True)
class TokenIO:
    """A compiled token graph's inputs and outputs, in the caller's order."""

    inputs: Sequence[str]
    outputs: Sequence[str]


DEBUG_ADAPTER = r"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", [] { return py::make_tuple(
        reinterpret_cast<std::uintptr_t>(circuits[0].run), circuits[0].aux); });
    m.def("evaluate", [](py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
                          std::uintptr_t ram) {
        py::array_t<Scalar> out({@ROWS@, circuits[0].width / @ROWS@});
        std::vector<int32_t> aux(std::max(1, circuits[0].aux));
        invoke(circuits[0].run, x.data(), reinterpret_cast<const uint8_t*>(ram),
               out.mutable_data(), aux.data());
        return out;
    });
}
"""


class CompiledGraph:
    """Cached token bank with Python and raw-buffer interfaces.

    Cache hits load compiled kernels; graph construction is deferred until
    compilation or diagnostic slot/liveness inspection.
    """

    def __init__(
        self,
        build_graphs: Callable[[], Mapping[str, TokenGraph]],
        *,
        tag: str,
        sources: tuple[str, ...],
        memory_size: int = 0x10000,
        dtype: str = "int32",
    ):
        if dtype not in ("int32", "float32"):
            raise ValueError("unsupported graph numeric type")
        self.dtype = np.dtype(dtype)
        self.build_graphs = build_graphs
        self.memory_size = memory_size
        native_files = tuple(str(p) for p in NATIVE_DIR.iterdir())
        key = f"graph:{tag}"
        code = hashlib.sha256(
            f"{build.source_digest(__name__, *sources, files=native_files)}|{torch.__version__}".encode()
        ).hexdigest()
        entry = build.manifest_read().get(key)
        module = None
        if (
            entry is not None
            and entry.get("sources") == code
            and entry.get("memory_size") == memory_size
            and entry.get("dtype", "int32") == dtype
        ):
            with build.artifact_lock(entry["digest"]):
                path = os.path.join(build.CACHE_DIR, f"graph_{entry['digest']}.cpp")
                if os.path.exists(path):
                    module = build.load_module(f"graph_gen_{entry['digest']}", entry["digest"], [path])
        if module is None:
            start = time.perf_counter()
            logger.info("building weight bank %s", tag)
            generated, source = self.generate_bank(build_graphs(), memory_size, dtype=dtype)
            logger.info(
                "generated %s: %d circuits, %.1f MiB source in %.1fs",
                tag, len(generated["model_ids"]), len(source.encode()) / 2**20,
                time.perf_counter() - start,
            )
            entry = {
                "kind": "graph",
                "sources": code,
                "memory_size": memory_size,
                "dtype": dtype,
                **generated,
            }
            with build.artifact_lock(entry["digest"]):
                path = os.path.join(build.CACHE_DIR, f"graph_{entry['digest']}.cpp")
                build.write_atomic(path, source.encode())
                module = build.load_module(f"graph_gen_{entry['digest']}", entry["digest"], [path])
                build.manifest_put(key, entry)
        self.graphs = {name: TokenIO(*io) for name, io in entry["tokens"].items()}
        self.digest, self.f_bits = entry["digest"], entry["f_bits"]
        self.source_code = code
        self.n_layers, self.d_model = entry["n_layers"], entry["d_model"]
        self.expert_names = entry["expert_names"]
        self.model_ids = {(graph, expert): i for graph, expert, i in entry["model_ids"]}
        self.engine = module.Engine()
        self._debug_modules = {}
        import weakref

        owner = weakref.proxy(self)
        self.engine.set_debug_loader(lambda trace: owner.debug_kernel(trace))

    @staticmethod
    def generate_bank(graphs: Mapping[str, TokenGraph], memory_size: int, *, dtype="int32") -> tuple[dict, str]:
        """Compile every graph and expert to weights and emit the engine's
        source; what the cache records of it."""
        model_ids, models = [], []
        for name, token in graphs.items():
            model_ids.append((name, None, len(models)))
            full = GraphFolder(
                token.program_graph, root_names=list(token.outputs), structural_reads=True
            ).fold({}, label=name)
            models.append(
                describe(
                    token, full.program_graph, list(full.dim_of_name.values()), dtype=dtype
                )
            )
            if token.experts:
                # Structural folds: each memory read becomes one engine-served
                # read (graph/fold.py), its annotated bits served with it.
                folder = GraphFolder(
                    token.program_graph, root_names=list(token.outputs), structural_reads=True
                )
                for expert, pins in token.experts.items():
                    fold = folder.fold(dict(pins), label=expert)
                    model_ids.append((name, expert, len(models)))
                    models.append(
                        describe(
                            token,
                            fold.program_graph,
                            list(fold.dim_of_name.values()),
                            dtype=dtype,
                        )
                    )
        source, f_bits = generate_graph_source(models, memory_size)
        source = f"// PyTorch {torch.__version__}\n" + source
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        return {
            "digest": digest,
            "f_bits": f_bits,
            "model_ids": model_ids,
            "n_layers": models[0]["n_layers"],
            "d_model": models[0]["d_model"],
            "expert_names": list(next(iter(graphs.values())).experts),
            "tokens": {name: (list(t.inputs), list(t.outputs)) for name, t in graphs.items()},
        }, source.replace("ENGINE_T", f"GraphEngine_{digest}")

    @functools.cached_property
    def primary_layout(self):
        """The primary graph's full model's construction info."""
        token = next(iter(self.build_graphs().values()))
        fold = GraphFolder(
            token.program_graph, root_names=list(token.outputs), structural_reads=True
        ).fold({}, label="primary")
        return construct(fold.program_graph)[2]

    @property
    def slot_of(self) -> dict:
        return self.primary_layout.slot_of

    @property
    def lives(self) -> list:
        return self.primary_layout.lives

    def model_id(self, graph: str, expert: str | None = None) -> int:
        return self.model_ids[graph, expert]

    def debug_kernel(self, trace: bool):
        if trace not in self._debug_modules:
            key = f"graph_debug:{self.digest}:{trace}"
            entry = build.manifest_read().get(key)
            if entry is not None and entry.get("code") == self.source_code:
                with build.artifact_lock(entry["digest"]):
                    path = Path(build.CACHE_DIR) / f"graph_{entry['digest']}_debug.cpp"
                    if path.exists():
                        self._debug_modules[trace] = build.load_module(
                            f"graph_debug_{entry['digest']}", entry["digest"], [str(path)]
                        )
                        return self._debug_modules[trace].kernel()
            token = next(iter(self.build_graphs().values()))
            fold = GraphFolder(
                token.program_graph, root_names=list(token.outputs), structural_reads=True
            ).fold({}, label="primary")
            model = describe(
                token, fold.program_graph, list(fold.dim_of_name.values()), dtype=self.dtype.name
            )
            model["fraction_bits"] = self.f_bits
            adapter = DEBUG_ADAPTER.replace("@ROWS@", str(model["n_layers"] + 1 if trace else 1))
            source = bank_source(
                [model], adapter, memory_size=self.memory_size, trace=trace, ablation=True
            )
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            with build.artifact_lock(digest):
                path = Path(build.CACHE_DIR) / f"graph_{digest}_debug.cpp"
                build.write_atomic(path, source.encode())
                self._debug_modules[trace] = build.load_module(
                    f"graph_debug_{digest}", digest, [str(path)]
                )
                build.manifest_put(key, {"kind": "graph_debug", "digest": digest}, code=self.source_code)
        return self._debug_modules[trace].kernel()

    def write(self, addr: int, value: int) -> None:
        self.engine.write(addr, value)

    def step(self, inputs, model: int = 0) -> np.ndarray:
        return self.engine.step(model, np.asarray(inputs, dtype=self.dtype))

    def abi(self) -> tuple:
        """(engine, write, step, trace, trace rows, trace width) for native
        callers; this object must outlive them."""
        return self.engine.abi()
