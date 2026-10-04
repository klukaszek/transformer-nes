"""Compile constructed transformer graphs with PyTorch Inductor.

The shared Torch compiler expands the actual attention and ReGLU matrices.
Structural memory reads are licensed by the graph fold before scheduling;
Inductor specializes constants and routing. Execution uses raw buffers and
one fixed byte image, with no framework dispatch or growing KV cache.

An engine hosts one or more token graphs over one memory (the PPU's:
the tile slot, and the sprite unit's evaluation and fetch).  Each comes
as its full model plus optional folded experts (graph/fold.py): the
graph partially evaluated under pinned conditions, rebuilt with its
memory reads served structurally (one exact read in place of the
grouped attention fabric), scheduled and constructed as its own smaller
weights.  Model 0 is the primary graph's full model and its experts
follow; then each further graph's.  The caller picks the model per
token and must only pick an expert whose pins hold.

The engine exposes per-token calls to Python (``write``, ``step``) and
the same calls as raw C function pointers (``abi``) so native code can
run the transformer with no Python in the loop:

    void write(void* engine, int addr, int value);
    void step(void* engine, int model, const int* inputs, int* outputs);
    void trace(void* engine, const int* inputs, int* residual);

``inputs``/``outputs`` follow the model's graph's named order (an input a
model folded away is ignored).  ``trace`` runs model 0 on a copy of the engine
and writes its residual stream after the embedding and after each layer
(the X-ray, nes/xray.py); ``ablate`` forces one of model 0's residual
slots to a constant after the layer that computes it.
model/inference.py is the float64 reference these engines agree with.
"""

from __future__ import annotations

import functools
import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from transformer_nes.files import write_atomic
from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    ProgramGraph,
    project_program_graph,
)
from transformer_nes.graph.fold import GraphFolder
from transformer_nes.model.build import (
    CACHE_DIR,
    artifact_lock,
    load_module,
    manifest_put,
    manifest_read,
)
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.sources import source_digest
from transformer_nes.model.torch_codegen import bank_source
from transformer_nes.model.weights import build_model


def construct(program_graph: ProgramGraph):
    schedule = asap_schedule(program_graph, log=lambda _: None)
    return build_model(program_graph=program_graph, schedule=schedule)


def generate_graph_source(models: list[dict], memory_size: int = 0x10000) -> tuple[str, int]:
    from pathlib import Path

    f_bits = max(m["fraction_bits"] for m in models)
    for model in models:
        model["fraction_bits"] = f_bits
    table = ["struct Token { int n_in, n_out; };", "static const Token tokens[] = {"]
    table.extend(f"{{{len(m['input_slots'])}, {len(m['output_slots'])}}}," for m in models)
    table.append("};")
    adapter = Path(__file__).with_name("native").joinpath("graph.cpp").read_text()
    for key, value in {
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
class TokenGraph:
    """A token type an engine runs: its graph, the caller's input and
    output order for it, and its folded experts (name -> pins: persist
    dim name -> constant)."""

    program_graph: ProgramGraph
    inputs: Sequence[str]
    outputs: Sequence[str]
    experts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def project(self, outputs: Sequence[str]) -> TokenGraph:
        """Compile only the dependency closure of selected readouts."""
        if not set(outputs) <= set(self.outputs):
            raise ValueError("projection must select existing token outputs")
        dims = {d.name: d for d in self.program_graph.all_dims}
        readout = Expression({dims[name]: 1e-6 for name in outputs})
        projected = project_program_graph(self.program_graph, {"READOUT": readout})
        return replace(self, program_graph=projected, outputs=tuple(outputs))


@dataclass(frozen=True)
class TokenIO:
    """A compiled token graph's inputs and outputs, in the caller's order."""

    inputs: Sequence[str]
    outputs: Sequence[str]


DEBUG_ADAPTER = r"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", [] { return py::make_tuple(
        reinterpret_cast<std::uintptr_t>(circuits[0].run), circuits[0].aux); });
    m.def("evaluate", [](py::array_t<int, py::array::c_style | py::array::forcecast> x,
                          std::uintptr_t ram) {
        py::array_t<int> out({@ROWS@, circuits[0].width / @ROWS@});
        std::vector<int32_t> aux(std::max(1, circuits[0].aux));
        invoke(circuits[0].run, x.data(), reinterpret_cast<const uint8_t*>(ram),
               out.mutable_data(), aux.data());
        return out;
    });
}
"""


class CompiledGraph:
    """Token graphs (and folded experts of them) compiled to weights and
    then to one native engine over one memory.  The first graph is the
    primary: its full model is model 0 (the one ``trace`` and ``ablate``
    see, described by ``slot_of``/``lives``) and its experts, named by
    ``expert_names``, are models 1..n; ``model_id`` finds any model.

    `build_graphs` builds the token graphs.  The engine is cached under
    `tag`, keyed by the code of `sources` (model/sources.py): a hit loads it
    without building any graph, and only ``slot_of``/``lives`` build the
    primary graph again (the X-ray's, on first use).  ``graphs`` gives
    each token graph's inputs and outputs, in the caller's order."""

    def __init__(
        self,
        build_graphs: Callable[[], Mapping[str, TokenGraph]],
        *,
        tag: str,
        sources: tuple[str, ...],
        memory_size: int = 0x10000,
    ):
        self.build_graphs = build_graphs
        self.memory_size = memory_size
        from pathlib import Path

        native_files = tuple(str(p) for p in Path(__file__).with_name("native").iterdir())
        key = f"graph:{tag}"
        code = hashlib.sha256(
            f"{source_digest(__name__, *sources, files=native_files)}|{torch.__version__}".encode()
        ).hexdigest()
        entry = manifest_read().get(key)
        module = None
        if (
            entry is not None
            and entry.get("sources") == code
            and entry.get("memory_size") == memory_size
        ):
            with artifact_lock(entry["digest"]):
                path = os.path.join(CACHE_DIR, f"graph_{entry['digest']}.cpp")
                if os.path.exists(path):
                    module = load_module(f"graph_gen_{entry['digest']}", entry["digest"], [path])
        if module is None:
            generated, source = self._generate(build_graphs(), memory_size)
            entry = {
                "kind": "graph",
                "sources": code,
                "memory_size": memory_size,
                **generated,
            }
            with artifact_lock(entry["digest"]):
                path = os.path.join(CACHE_DIR, f"graph_{entry['digest']}.cpp")
                write_atomic(path, source.encode())
                module = load_module(f"graph_gen_{entry['digest']}", entry["digest"], [path])
                manifest_put(key, entry)
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
        self.engine.set_debug_loader(lambda trace: owner._debug_kernel(trace))

    @staticmethod
    def _generate(graphs: Mapping[str, TokenGraph], memory_size: int) -> tuple[dict, str]:
        """Compile every graph and expert to weights and emit the engine's
        source; what the cache records of it."""
        model_ids, models = [], []
        for name, token in graphs.items():
            model_ids.append((name, None, len(models)))
            full = GraphFolder(
                token.program_graph, root_names=list(token.outputs), structural_reads=True
            ).fold({}, label=name)
            models.append(
                CompiledGraph._compile(
                    token, full.program_graph, list(full.dim_of_name.values()), full.union_lookup_of
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
                        CompiledGraph._compile(
                            token,
                            fold.program_graph,
                            list(fold.dim_of_name.values()),
                            fold.union_lookup_of,
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
    def _primary(self):
        """The primary graph's full model's construction info."""
        token = next(iter(self.build_graphs().values()))
        fold = GraphFolder(
            token.program_graph, root_names=list(token.outputs), structural_reads=True
        ).fold({}, label="primary")
        return construct(fold.program_graph)[2]

    @property
    def slot_of(self) -> dict:
        return self._primary.slot_of

    @property
    def lives(self) -> list:
        return self._primary.lives

    def model_id(self, graph: str, expert: str | None = None) -> int:
        return self.model_ids[graph, expert]

    @staticmethod
    def _compile(token: TokenGraph, program_graph, dims, union_lookup_of) -> dict:
        model, _idx, info = construct(program_graph)
        by_name = {}
        for d in dims:
            by_name.setdefault(d.name, d)
        inputs = {d.name: d for d in program_graph.all_dims if isinstance(d, InputDimension)}
        return {
            "model": model,
            # Integer coefficients preserve integer residuals. Fractional
            # projections need the same half-unit encoding as the CPU.
            "fraction_bits": int(any(
                (weight.weight != np.rint(weight.weight)).any()
                for weight in [
                    *model.ff_in, *model.ff_out,
                    *(attn.out_proj for attn in model.attn),
                ]
            )),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "one_slot": info.slot_of[program_graph.one],
            "input_slots": [
                info.slot_of.get(inputs[n], -1) if n in inputs else -1 for n in token.inputs
            ],
            "output_slots": [info.slot_of[by_name[n]] for n in token.outputs],
            "union_lookup_of": union_lookup_of,
        }

    def _debug_kernel(self, trace: bool):
        if trace not in self._debug_modules:
            from pathlib import Path

            key = f"graph_debug:{self.digest}:{trace}"
            entry = manifest_read().get(key)
            if entry is not None and entry.get("code") == self.source_code:
                with artifact_lock(entry["digest"]):
                    path = Path(CACHE_DIR) / f"graph_{entry['digest']}_debug.cpp"
                    if path.exists():
                        self._debug_modules[trace] = load_module(
                            f"graph_debug_{entry['digest']}", entry["digest"], [str(path)]
                        )
                        return self._debug_modules[trace].kernel()
            token = next(iter(self.build_graphs().values()))
            fold = GraphFolder(
                token.program_graph, root_names=list(token.outputs), structural_reads=True
            ).fold({}, label="primary")
            model = self._compile(
                token, fold.program_graph, list(fold.dim_of_name.values()), fold.union_lookup_of
            )
            model["fraction_bits"] = self.f_bits
            adapter = DEBUG_ADAPTER.replace("@ROWS@", str(model["n_layers"] + 1 if trace else 1))
            source = bank_source(
                [model], adapter, memory_size=self.memory_size, trace=trace, ablation=True
            )
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            with artifact_lock(digest):
                path = Path(CACHE_DIR) / f"graph_{digest}_debug.cpp"
                write_atomic(path, source.encode())
                self._debug_modules[trace] = load_module(
                    f"graph_debug_{digest}", digest, [str(path)]
                )
                manifest_put(key, {"kind": "graph_debug", "digest": digest}, code=self.source_code)
        return self._debug_modules[trace].kernel()

    def write(self, addr: int, value: int) -> None:
        self.engine.write(addr, value)

    def step(self, inputs, model: int = 0) -> np.ndarray:
        return self.engine.step(model, np.asarray(inputs, dtype=np.int32))

    def abi(self) -> tuple:
        """(engine, write, step, trace, trace rows, trace width) for native
        callers; this object must outlive them."""
        return self.engine.abi()
