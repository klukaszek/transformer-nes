"""Compile any constructed graph into a native transformer engine.

The same emission the 6502 engine uses (``plan_model``/``emit_layer`` in
model/engine_gen.py): each layer's licensed attention heads, fused
direct maps and ReGLU block as straight-line exact fixed-point C++.
Memory heads -- grouped exact-address reads over WRITE tokens -- are
served by the nearest-written-address structure, so a WRITE token is one
structure insert and the attention state stays O(1) in the log.

An engine hosts model 0, the full graph, plus optional folded experts
(graph/fold.py): the graph partially evaluated under pinned conditions,
rebuilt with its memory reads served structurally (one exact read in
place of the grouped attention fabric), scheduled and constructed as
its own smaller weights.  All models share one memory; the caller picks
the model per token and must only pick an expert whose pins hold.

The engine exposes per-token calls to Python (``write``, ``step``) and
the same calls as raw C function pointers (``abi``) so native code can
run the transformer with no Python in the loop:

    void write(void* engine, int addr, int value);
    void step(void* engine, int model, const int* inputs, int* outputs);
    void trace(void* engine, const int* inputs, int* residual);

``inputs``/``outputs`` follow the caller's named order (an input a model
folded away is ignored).  ``trace`` runs model 0 on a copy of the engine
and writes its residual stream after the embedding and after each layer
(the X-ray, nes/xray.py); ``ablate`` forces one of model 0's residual
slots to a constant after the layer that computes it.
model/inference.py is the float64 reference these engines agree with.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from transformer_nes.cpu6502.backends.weights.run_protocol import mem_query_cpp
from transformer_nes.graph.core import InputDimension, ProgramGraph
from transformer_nes.graph.fold import GraphFolder
from transformer_nes.model.engine_gen import (
    CACHE_DIR,
    FIXED_POINT_HELPERS,
    INCLUDES,
    EmitterContext,
    emit_layer,
    load_module,
    manifest_put,
    manifest_read,
    plan_model,
)
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.weights import build_model


def construct(program_graph: ProgramGraph):
    schedule = asap_schedule(
        program_graph.input_tokens,
        program_graph.output_tokens,
        program_graph=program_graph,
        log=lambda _: None,
    )
    return build_model(program_graph=program_graph, schedule=schedule)


def generate_graph_source(models: list[dict], memory_size: int = 0x10000) -> tuple[str, int]:
    """models: [{model, d_model, one_slot, input_slots, output_slots,
    union_lookup_of}], model 0 the union.

    The residual is a local array of each step with every layer inlined
    and every index constant, so the compiler keeps the stream in
    registers and drops dead erases.  The fixed-point scale (F=1) only
    exists for +-0.5 coefficients; a source without them runs on plain
    integers (F=0), where the scale checks compile away.  Returns the
    source and its F."""
    source = _generate(models, memory_size, f_bits=0)
    # One "half(" is the helper's own definition; any other is a use.
    if source.count("half(") == 1:
        return source, 0
    return _generate(models, memory_size, f_bits=1), 1


def _generate(models: list[dict], memory_size: int, *, f_bits: int) -> str:
    ctx = EmitterContext(elem="int", E="", ram_size=memory_size)
    groups = -(-memory_size // 4096)
    plans = [
        plan_model(ctx, m["model"], f"model {k}", None if k == 0 else m["union_lookup_of"])
        for k, m in enumerate(models)
    ]
    n_regs = max(1, len(ctx.reg_of))
    n_in = len(models[0]["input_slots"])
    n_out = len(models[0]["output_slots"])
    lines = [INCLUDES, "", "struct ENGINE_T {"]
    emit = lines.append
    emit(f"    static constexpr int F = {f_bits};")
    emit("    static constexpr int SCALE = 1 << F;")
    emit(FIXED_POINT_HELPERS.replace("ELEM", "int"))
    emit(mem_query_cpp("long long", groups))
    emit("    // The same writes as a flat image: folded experts' reads are")
    emit("    // served from it (value and bits), not from the attention fabric.")
    emit(f"    static constexpr int MEMORY = {memory_size};")
    emit("    unsigned char ram[MEMORY] = {};")
    emit(f"    bool reg_any[{n_regs}] = {{}};")
    emit(f"    long long reg_s[{n_regs}] = {{}};")
    emit(f"    int reg_v0[{n_regs}] = {{}}, reg_v1[{n_regs}] = {{}};")
    emit("    long long seq = -1;")
    emit(f"    static constexpr int N_MODELS = {len(models)}, N_IN = {n_in}, N_OUT = {n_out};")
    emit("")
    # Each model's layers inline into its step: one straight-line body the
    # compiler can keep in registers across layer boundaries.
    for k, model_plans in enumerate(plans):
        for plan in model_plans:
            layer = emit_layer(plan, f"_m{k}", "int* x")
            layer[0] = layer[0].replace("    void ", "    __attribute__((always_inline)) void ", 1)
            lines.extend(layer)
    emit("")
    emit("    // A WRITE token: one insert into the memory structure.")
    emit("    void write(int addr, int value) {")
    emit("        ++seq;")
    emit("        if ((unsigned)addr >= (unsigned)MEMORY)")
    emit('            throw std::out_of_range("write outside the engine memory");')
    emit("        const int group = addr >> 12, wl = addr & 0xFFF;")
    emit("        mem_val[group][wl] = value;")
    emit("        mem_seq[group][wl] = seq;")
    emit("        mem_bits[group][wl >> 6] |= 1ULL << (wl & 63);")
    emit("        ram[addr] = (unsigned char)value;")
    emit("    }")
    # Ablation (the X-ray): model 0 forces one residual slot to a value
    # right after the layer that computes it.  ablate_row -1 = off.
    emit("    int ablate_slot = 0, ablate_row = -1, ablate_value = 0;")
    def forward(k: int, indent: str, snap: bool) -> None:
        """Model k's embedding and layer stack; model 0 also applies the
        ablation, and with `snap` records the stream after each row."""
        m, d = models[k], models[k]["d_model"]
        emit(f"{indent}int x[{d}] = {{}};")
        emit(f"{indent}x[{m['one_slot']}] = SCALE;")
        for i, slot in enumerate(m["input_slots"]):
            if slot >= 0:
                emit(f"{indent}x[{slot}] = in[{i}] * SCALE;")
        if snap:
            emit(f"{indent}std::memcpy(snap, x, sizeof x);")
        for row, plan in enumerate(plans[k], start=1):
            emit(f"{indent}layer_{plan['li']}_m{k}(x);")
            if k == 0:
                emit(f"{indent}if (ablate_row == {row}) x[ablate_slot] = ablate_value;")
            if snap:
                emit(f"{indent}std::memcpy(snap + {row * d}, x, sizeof x);")

    emit("    // Any other token, through model m: embed, run its stack, read out.")
    emit("    void step(int m, const int* in, int* out) {")
    emit("        ++seq;")
    emit("        switch (m) {")
    for k, m in enumerate(models):
        emit(f"        case {k}: {{")
        forward(k, "            ", snap=False)
        for i, slot in enumerate(m["output_slots"]):
            emit(f"            out[{i}] = (int)shr2(x[{slot}]);")
        emit("            return;")
        emit("        }")
    emit("        }")
    emit('        throw std::out_of_range("no such model");')
    emit("    }")
    emit(f"    static constexpr int TRACE_ROWS = {len(plans[0]) + 1}, "
         f"TRACE_D = {models[0]['d_model']};")
    emit("    // Model 0, the residual stream after each row written to snap.")
    emit("    void trace(const int* in, int* snap) {")
    emit("        ++seq;")
    forward(0, "        ", snap=True)
    emit("    }")
    emit("};")
    emit("")
    emit("static void abi_write(void* e, int addr, int value) {")
    emit("    static_cast<ENGINE_T*>(e)->write(addr, value);")
    emit("}")
    emit("static void abi_step(void* e, int m, const int* in, int* out) {")
    emit("    static_cast<ENGINE_T*>(e)->step(m, in, out);")
    emit("}")
    emit("// Traced on a copy, so the running engine never sees the extra token.")
    emit("static void abi_trace(void* e, const int* in, int* snap) {")
    emit("    std::make_unique<ENGINE_T>(*static_cast<ENGINE_T*>(e))->trace(in, snap);")
    emit("}")
    emit("")
    emit("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    emit('    py::class_<ENGINE_T>(m, "Engine")')
    emit("        .def(py::init<>())")
    emit('        .def("write", &ENGINE_T::write)')
    emit(
        '        .def("step", [](ENGINE_T& e, int model, py::array_t<int, '
        "py::array::c_style | py::array::forcecast> in) {"
    )
    emit("            if (in.size() != ENGINE_T::N_IN)")
    emit('                throw std::invalid_argument("wrong input count");')
    emit("            py::array_t<int> out(ENGINE_T::N_OUT);")
    emit("            e.step(model, in.data(), out.mutable_data());")
    emit("            return out;")
    emit("        })")
    emit('        .def("ablate", [](ENGINE_T& e, int slot, int row, int value) {')
    emit("            e.ablate_slot = slot;")
    emit("            e.ablate_row = row;")
    emit("            e.ablate_value = value * ENGINE_T::SCALE;")
    emit("        }, py::arg(\"slot\") = 0, py::arg(\"row\") = -1, py::arg(\"value\") = 0)")
    emit('        .def("abi", [](ENGINE_T& e) {')
    emit("            return py::make_tuple(reinterpret_cast<std::uintptr_t>(&e),")
    emit("                                  reinterpret_cast<std::uintptr_t>(&abi_write),")
    emit("                                  reinterpret_cast<std::uintptr_t>(&abi_step),")
    emit("                                  reinterpret_cast<std::uintptr_t>(&abi_trace),")
    emit("                                  ENGINE_T::TRACE_ROWS, ENGINE_T::TRACE_D);")
    emit("        });")
    emit("}")
    return "\n".join(lines) + "\n"


class CompiledGraph:
    """A graph (and folded experts of it) compiled to weights and then to
    a native engine.  ``experts`` maps a name to its pins (persist dim
    name -> constant); their model indices follow in order after 0."""

    def __init__(
        self,
        program_graph: ProgramGraph,
        inputs: list[str],
        outputs: list[str],
        *,
        tag: str,
        experts: dict[str, dict[str, float]] | None = None,
        memory_size: int = 0x10000,
    ):
        self.inputs, self.outputs = list(inputs), list(outputs)
        self.expert_names = list(experts or {})
        models = [self._compile(program_graph, program_graph.all_dims, None)]
        # The full model's residual layout: the X-ray labels its stream.
        self.slot_of, self.lives = models[0]["info"].slot_of, models[0]["info"].lives
        if experts:
            # Structural folds: each memory read becomes one engine-served
            # read (graph/fold.py), its annotated bits served with it.
            folder = GraphFolder(program_graph, root_names=outputs, structural_reads=True)
            for name, pins in experts.items():
                fold = folder.fold(pins, label=name)
                models.append(
                    self._compile(
                        fold.program_graph,
                        list(fold.dim_of_name.values()),
                        fold.union_lookup_of,
                    )
                )
        self.layers = [m["n_layers"] for m in models]
        self.widths = [m["d_model"] for m in models]
        self.n_layers, self.d_model = self.layers[0], self.widths[0]
        source, self.f_bits = generate_graph_source(models, memory_size)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.digest = digest
        source = source.replace("ENGINE_T", f"GraphEngine_{digest}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"graph_{digest}.cpp")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(source)
        module = load_module(f"graph_gen_{digest}", digest, [path])
        key = f"graph:{tag}"
        if manifest_read().get(key, {}).get("digest") != digest:
            manifest_put(key, {"kind": "graph", "digest": digest})
        self.engine = module.Engine()

    def _compile(self, program_graph, dims, union_lookup_of) -> dict:
        model, _tokens, _idx, info = construct(program_graph)
        by_name = {}
        for d in dims:
            by_name.setdefault(d.name, d)
        inputs = {d.name: d for d in program_graph.all_dims if isinstance(d, InputDimension)}
        return {
            "model": model,
            "info": info,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "one_slot": info.slot_of[program_graph.one],
            "input_slots": [
                info.slot_of.get(inputs[n], -1) if n in inputs else -1 for n in self.inputs
            ],
            "output_slots": [info.slot_of[by_name[n]] for n in self.outputs],
            "union_lookup_of": union_lookup_of,
        }

    def write(self, addr: int, value: int) -> None:
        self.engine.write(addr, value)

    def step(self, inputs, model: int = 0) -> np.ndarray:
        return self.engine.step(model, np.asarray(inputs, dtype=np.int32))

    def abi(self) -> tuple:
        """(engine, write, step, trace, trace rows, trace width) for native
        callers; this object must outlive them."""
        return self.engine.abi()
