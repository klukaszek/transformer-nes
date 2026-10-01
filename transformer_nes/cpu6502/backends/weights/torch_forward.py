"""Run the CPU engine's expert forward passes in PyTorch.

The native run loop (model/engine_gen.py) keeps dispatch, budgets and
device pauses.  An engine built with the forward hook embeds an expert's
inputs into the residual stream, then hands the stream to the hook, which
runs the expert's layers in PyTorch and writes the result back.

Every layer of every expert is model/expert_runner.py:layer with different
weights, so it is compiled once by AOTInductor, with every size dynamic,
into a few shared libraries (one per count of layers per call, up to 8,
and one for a layer with half coefficients), cached by the layer's source
and the torch version.  An expert is then only data: its layers' weight
arrays.  The hook (_HOOK_SOURCE) is native code that calls the libraries
through their minimal ArrayRef interface on the engine's own residual and
RAM, with no Python per forward.

The libraries build once and the experts load in the background; until
an expert is loaded, the hook declines it and the engine runs it itself.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import inspect
import logging
import os
import subprocess
import sys
import threading
import zipfile

import torch

from transformer_nes.cpu6502.backends.weights.runner import window_folds
from transformer_nes.files import write_atomic
from transformer_nes.model import expert_runner
from transformer_nes.model.engine_gen import CACHE_DIR, SEQ_HOOK_IDS
from transformer_nes.model.expert_runner import HALVES, INT, WEIGHTS, Expert
from transformer_nes.model.export import expert_artifact

logger = logging.getLogger(__name__)
SCALE_BITS = 1  # the engine's fixed point (F) and the checkpoint's
SLOTS = SEQ_HOOK_IDS + 8192  # hook ids: the core's, then the windows'
# The libraries, as (layers per call, with half coefficients).  A call
# costs about as much as a layer and experts have few layers (3-4 on
# average), so one library per count up to 8 makes most experts one call.
LIBRARIES = (*((n, False) for n in range(1, 9)), (1, True))
# A layer's loops are a few elements long: one thread, scalar code (vector
# code spends its time on partial vectors), intermediates planned into few
# allocations, and the minimal ArrayRef interface, which this torch emits
# only with stack allocation allowed.
INDUCTOR_CONFIGS = {
    "cpp.threads": 1,
    "cpp.vec_isa_ok": False,
    "memory_planning": True,
    "aot_inductor.allow_stack_allocation": True,
    "aot_inductor.use_minimal_arrayref_interface": True,
}

_HOOK_SOURCE = r"""
#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <atomic>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

using torch::aot_inductor::ArrayRefTensor;
using I32 = ArrayRefTensor<int32_t>;
using U8 = ArrayRefTensor<uint8_t>;
template <size_t N, class... Weights>
struct ArgsOf : ArgsOf<N - 1, I32, Weights...> {};
template <class... Weights>
struct ArgsOf<0, Weights...> { using type = std::tuple<I32, U8, Weights...>; };
// A library's inputs: the residual, RAM, then each layer's WEIGHTS (and
// HALVES); its outputs: the residual, the addresses read.
template <size_t Arrays> using Inputs = typename ArgsOf<Arrays>::type;
using Outputs = std::tuple<I32, I32>;

// A weight array, with row-major strides of its own (torch's can differ
// where a size is 0 or 1, and the ArrayRef interface checks them).
struct Weight {
    torch::Tensor data;
    int64_t strides[2];
    explicit Weight(torch::Tensor t) : data(t.contiguous()), strides{t.dim() == 2 ? t.size(1) : 1, 1} {}
    I32 view() const {
        return I32({data.data_ptr<int32_t>(), (size_t)data.numel()}, {data.sizes().data(), data.sizes().size()},
                   {strides, (size_t)data.dim()}, aoti_torch_device_type_cpu(), 0);
    }
};

// A library, called the bare-metal way AOTInductor offers: one model made
// once, run through the minimal ArrayRef interface.
template <size_t Arrays>
struct Library {
    AOTInductorModelHandle model = nullptr;
    AOTIRuntimeError (*run_model)(AOTInductorModelHandle, const Inputs<Arrays>&, Outputs&) = nullptr;

    void load(const std::string& path) {
        void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(library, "cannot load ", path, ": ", dlerror());
        auto create = reinterpret_cast<AOTIRuntimeError (*)(AOTInductorModelHandle*, AOTInductorConstantMapHandle)>(
            dlsym(library, "AOTInductorModelCreate"));
        run_model = reinterpret_cast<decltype(run_model)>(dlsym(library, "AOTInductorModelRunMinimalArrayrefInterface"));
        TORCH_CHECK(create && run_model, path, " has no minimal ArrayRef interface");
        TORCH_CHECK(create(&model, nullptr) == AOTI_RUNTIME_SUCCESS, "cannot create ", path);
    }
    template <size_t... N>
    bool run(const I32& x, const U8& ram, const std::vector<Weight>& w, Outputs& out,
             std::index_sequence<N...>) const {
        return run_model(model, Inputs<Arrays>{x, ram, w[N].view()...}, out) == AOTI_RUNTIME_SUCCESS;
    }
    bool run(const I32& x, const U8& ram, const std::vector<Weight>& w, Outputs& out) const {
        TORCH_CHECK(w.size() == Arrays, "a call with the wrong number of weight arrays");
        return run(x, ram, w, out, std::make_index_sequence<Arrays>());
    }
};

// An expert: its layers as library calls, each (library, weight arrays).
struct Expert {
    std::vector<std::pair<int, std::vector<Weight>>> calls;
    int64_t width;
};

// The libraries, in LIBRARIES' order, and each one's arrays per layer;
// read_kind's place among a layer's (all generated from the Python side).
using Libraries = std::tuple<@LIBRARIES@>;
static constexpr size_t n_libraries = std::tuple_size_v<Libraries>;
static constexpr size_t layer_arrays[n_libraries] = {@LAYER_ARRAYS@};
static constexpr size_t read_kind = @READ_KIND@;

// Slots are filled from another thread while the engine runs: a forward
// sees an expert or nothing, and declines nothing (or anything before the
// libraries load).  A hook lives as long as the process.
struct Hook {
    std::vector<std::atomic<Expert*>> slots;
    Libraries libraries;
    std::atomic<bool> loaded{false};
    uint8_t* ram;
    const bool* xray_on;
    unsigned int* xray_reads;
    long calls = 0, declined = 0;
    explicit Hook(int n) : slots(n) {}

    // Runs std::get<library>(libraries).
    template <size_t... L>
    bool run(int library, const I32& x, const U8& ram, const std::vector<Weight>& w, Outputs& out,
             std::index_sequence<L...>) const {
        bool ok = false;
        ((library == (int)L && (ok = std::get<L>(libraries).run(x, ram, w, out), true)) || ...);
        return ok;
    }
    bool run(int library, const I32& x, const U8& ram, const std::vector<Weight>& w, Outputs& out) const {
        return run(library, x, ram, w, out, std::make_index_sequence<n_libraries>());
    }
    // Counts each address a layer reads once, as the engine's X-ray does
    // (heads sharing a query share one read).  `reads` holds each layer's
    // heads' addresses in turn, one per row of its read_kind.
    void count_reads(const int32_t* reads, const std::vector<Weight>& weights, size_t stride) const {
        for (size_t layer = 0; layer < weights.size(); layer += stride) {
            const int64_t heads = weights[layer + read_kind].data.size(0);
            for (int64_t i = 0; i < heads; ++i) {
                const int32_t address = reads[i];
                bool seen = address < 0;
                for (int64_t j = 0; j < i && !seen; ++j) seen = reads[j] == address;
                if (!seen) ++xray_reads[address];
            }
            reads += heads;
        }
    }
    template <size_t... L>
    void load(const std::vector<std::string>& paths, std::index_sequence<L...>) {
        TORCH_CHECK(paths.size() == n_libraries, "expected ", n_libraries, " libraries");
        (std::get<L>(libraries).load(paths[L]), ...);
        loaded.store(true, std::memory_order_release);
    }
};

static bool forward(void* ctx, int id, int* x) {
    static constexpr int64_t memory_size = 65536, unit = 1;
    auto& h = *static_cast<Hook*>(ctx);
    const Expert* expert = h.loaded.load(std::memory_order_acquire)
        ? h.slots[id].load(std::memory_order_acquire) : nullptr;
    if (!expert) {
        ++h.declined;
        return false;
    }
    ++h.calls;
    const int32_t cpu = aoti_torch_device_type_cpu();
    const I32 residual({x, (size_t)expert->width}, {&expert->width, 1}, {&unit, 1}, cpu, 0);
    const U8 ram({h.ram, (size_t)memory_size}, {&memory_size, 1}, {&unit, 1}, cpu, 0);
    Outputs out;
    for (const auto& [library, weights] : expert->calls) {
        TORCH_CHECK(h.run(library, residual, ram, weights, out), "expert ", id, " failed");
        std::memcpy(x, std::get<0>(out).data(), sizeof(int32_t) * expert->width);
        if (*h.xray_on) h.count_reads(std::get<1>(out).data(), weights, layer_arrays[library]);
    }
    return true;
}

std::tuple<uint64_t, uint64_t> make_hook(int64_t slots, uint64_t ram, uint64_t xray_on,
                                         uint64_t xray_reads) {
    auto* h = new Hook((int)slots);
    for (auto& slot : h->slots) slot.store(nullptr);
    h->ram = reinterpret_cast<uint8_t*>(ram);
    h->xray_on = reinterpret_cast<const bool*>(xray_on);
    h->xray_reads = reinterpret_cast<unsigned int*>(xray_reads);
    return {reinterpret_cast<uint64_t>(&forward), reinterpret_cast<uint64_t>(h)};
}

void load_libraries(uint64_t ctx, const std::vector<std::string>& paths) {
    reinterpret_cast<Hook*>(ctx)->load(paths, std::make_index_sequence<n_libraries>());
}

// Puts an expert in a slot (width -1: none), deleting the one it replaces
// -- so with the engine idle, unless the slot was empty.
void set_expert(uint64_t ctx, int64_t id,
                const std::vector<std::tuple<int64_t, std::vector<torch::Tensor>>>& calls, int64_t width) {
    Expert* expert = nullptr;
    if (width >= 0) {
        expert = new Expert{{}, width};
        for (const auto& [library, weights] : calls)
            expert->calls.emplace_back((int)library, std::vector<Weight>(weights.begin(), weights.end()));
    }
    delete reinterpret_cast<Hook*>(ctx)->slots.at(id).exchange(expert);
}

std::tuple<int64_t, int64_t> counts(uint64_t ctx) {
    const auto& h = *reinterpret_cast<Hook*>(ctx);
    return {h.calls, h.declined};
}
"""


def arrays_per_call(n_layers: int, halves: bool) -> int:
    """A library's weight arrays per call."""
    return n_layers * len(WEIGHTS + (HALVES if halves else ()))


@functools.cache
def _hook_module():
    from torch.utils.cpp_extension import load_inline

    generated = {
        "@LIBRARIES@": ", ".join(f"Library<{arrays_per_call(*library)}>" for library in LIBRARIES),
        "@LAYER_ARRAYS@": ", ".join(str(arrays_per_call(1, halves)) for _n, halves in LIBRARIES),
        "@READ_KIND@": str(WEIGHTS.index("read_kind")),
    }
    source = _HOOK_SOURCE
    for name, text in generated.items():
        source = source.replace(name, text)
    return load_inline(
        "transformer_nes_forward_hook", cpp_sources=source,
        functions=["make_hook", "load_libraries", "set_expert", "counts"], extra_cflags=["-O2"],
    )


def library_calls(layers) -> list[tuple[int, list[torch.Tensor]]]:
    """An expert's layers as library calls (index into LIBRARIES, weight
    arrays): runs of plain layers in as few calls as the libraries allow,
    each layer with half coefficients alone."""
    calls, run = [], []

    def flush():
        while run:
            n = max(k for k, halves in LIBRARIES if not halves and k <= len(run))
            calls.append((LIBRARIES.index((n, False)), [w for layer in run[:n] for w in layer.weights()]))
            del run[:n]

    for layer in layers:
        if layer.has_halves:
            flush()
            calls.append((LIBRARIES.index((1, True)), layer.weights()))
        else:
            run.append(layer)
    flush()
    return calls


class Layers(torch.nn.Module):
    """model/expert_runner.py:layer, in sequence, as AOTInductor compiles
    it: the residual, RAM and each layer's weights in; the residual and the
    addresses read out."""

    def forward(self, x, memory, *layers):
        reads = []
        for weights in layers:
            x, read = expert_runner.layer(x, memory, SCALE_BITS, *weights)
            reads.append(read)
        return x, torch.cat(reads)


def _example(n_layers: int, halves: bool) -> tuple[tuple, tuple]:
    """Inputs to trace `n_layers` layers with, and their dynamic sizes.
    The tracer takes equal example sizes for one size, so every size is
    distinct, including the derived ones (2 * reads + values, 2 * neurons)."""
    width = 101  # the residual's
    taken = {1, 2, width}

    def fresh(*derived) -> int:
        """The least size n with n and each derived(n) not yet taken."""
        n = 3
        while {n, *(f(n) for f in derived)} & taken:
            n += 1
        taken.update({n, *(f(n) for f in derived)})
        return n

    dynamic = torch.export.Dim.DYNAMIC
    names = WEIGHTS + (HALVES if halves else ())
    layers, shapes = [], []
    for _ in range(n_layers):
        reads = fresh()
        values = fresh(lambda v, r=reads: 2 * r + v)
        n_in, n_out, ff_in, ff_out = fresh(), fresh(), fresh(), fresh()
        neurons = fresh(lambda f: 2 * f)
        rows = 2 * reads + values
        size = {
            "in_slots": (n_in,), "w_in": (rows, n_in), "read_kind": (reads, 2),
            "out_slots": (n_out,), "w_out": (n_out, rows), "ff_in_slots": (ff_in,),
            "w_ff": (2 * neurons, ff_in), "ff_out_slots": (ff_out,), "w_down": (ff_out, neurons),
        }
        size |= {h: size["w" + h[1:]] for h in HALVES}
        layers.append(tuple(
            torch.randint(0, width if name.endswith("slots") else 3, size[name], dtype=INT)
            for name in names))
        shapes.append(tuple(
            {axis: dynamic for axis in range(len(size[name])) if (name, axis) != ("read_kind", 1)}
            for name in names))
    x = torch.randint(0, 3, (width,), dtype=INT)
    ram = torch.zeros(0x10000, dtype=torch.uint8)
    # Export takes forward's *layers as one tuple argument.
    return (x, ram, *layers), ({0: dynamic}, None, tuple(shapes))


@functools.cache
def library_path(n_layers: int, halves: bool) -> str:
    """A library, keyed by the layer's source and the compiler and its
    settings."""
    sources = (inspect.getsource(expert_runner), inspect.getsource(Layers), inspect.getsource(_example),
               repr(INDUCTOR_CONFIGS), torch.__version__, repr((n_layers, halves)))
    digest = hashlib.sha256("|".join(sources).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"layers_{digest}.so")


def compile_library(n_layers: int, halves: bool) -> str:
    """Compile a library and take its shared object out of the AOTInductor
    package (loading it directly skips an unzip per start).  Run in a fresh
    process: a process's first compile fixes the wrapper kind it emits."""
    path = library_path(n_layers, halves)
    example, dynamic = _example(n_layers, halves)
    with torch._inductor.config.patch(INDUCTOR_CONFIGS):
        program = torch.export.export(Layers().eval(), example, dynamic_shapes=dynamic)
        package = torch._inductor.aoti_compile_and_package(program, package_path=path + ".pt2")
    with zipfile.ZipFile(package) as archive:
        (library,) = [name for name in archive.namelist() if name.endswith(".so")]
        write_atomic(path, archive.read(library))
    os.remove(package)
    return path


def libraries() -> list[str]:
    """The libraries (LIBRARIES' order), compiled (in a fresh process) if
    missing; any other code's are deleted."""
    paths = [library_path(*library) for library in LIBRARIES]
    for stale in set(glob.glob(os.path.join(CACHE_DIR, "layers_*.so"))) - set(paths):
        os.remove(stale)
    if not all(map(os.path.exists, paths)):
        logger.info("compiling the layer libraries (AOTInductor, once)")
        subprocess.run([sys.executable, "-m", __name__], check=True)
    return paths


class TorchForwards:
    """`machine`'s expert forwards in PyTorch (machine: a
    Cpu6502TransformerMachine built forward_hook=True).  `wait` blocks
    until every expert is loaded, so a benchmark measures PyTorch alone."""

    def __init__(self, machine, *, wait: bool = False):
        if not machine.forward_hook:
            raise ValueError("the machine's engine was built without the forward hook")
        self.machine = machine
        self.hook = _hook_module()
        engine = machine.engine.engine
        fn, self.ctx = self.hook.make_hook(SLOTS, engine.ram_address(), *engine.xray_address())
        self.lock = threading.Lock()
        self.generation = 0  # of the window set
        self.windows = 0  # experts loaded from it
        self.ready = threading.Event()
        self.failure: BaseException | None = None
        engine.set_forward_hook(fn, self.ctx)
        windows = machine.seq_windows if machine.seq_key else ()
        threading.Thread(target=self._start, args=(windows, wait), daemon=True).start()
        if wait:
            self.wait()

    def attach_windows(self) -> None:
        """Serve the windows the machine has attached since (call between
        frames: replaced experts are deleted)."""
        with self.lock:
            self.generation += 1
            for j in range(self.windows):
                self.hook.set_expert(self.ctx, SEQ_HOOK_IDS + j, [], -1)
            self.windows = 0
        args = (self.machine.seq_windows, self.generation)
        threading.Thread(target=lambda: (self.ready.wait(), self._serve_windows(*args)),
                         daemon=True).start()

    def wait(self) -> None:
        self.ready.wait()
        if self.failure:
            raise RuntimeError("PyTorch forwards failed to start") from self.failure

    @property
    def counts(self) -> tuple[int, int]:
        """(forwards PyTorch ran, forwards it declined to the engine)."""
        return tuple(self.hook.counts(self.ctx))

    def _start(self, windows: tuple, build: bool) -> None:
        """Load the libraries and the experts: the core engine's, then the
        windows' (folding missing ones only if `build`; a game leaves that
        to its learner)."""
        try:
            self.hook.load_libraries(self.ctx, libraries())
            self._serve(dict(enumerate(self.machine.core_experts()[0])), self.generation)
            self._serve_windows(windows, self.generation, build=build)
            logger.info("PyTorch runs every expert's forwards")
        except Exception as error:
            logger.error("PyTorch forwards failed to start: %s", error)
            self.failure = error
        self.ready.set()

    def _serve_windows(self, windows: tuple, generation: int, *, build: bool = False) -> None:
        folds = window_folds(self.machine.pg, windows, self.machine.bcd, build=build) or ()
        self._serve(dict(enumerate(folds, SEQ_HOOK_IDS)), generation)

    def _serve(self, experts: dict, generation: int) -> None:
        """Put each expert (by hook id) in its slot as library calls, unless
        its window set has been replaced meanwhile."""
        for i, folded in experts.items():
            tensors, meta = expert_artifact(folded, 0)
            calls = library_calls(Expert({k: torch.from_numpy(v) for k, v in tensors.items()}, meta).layers)
            with self.lock:
                if i >= SEQ_HOOK_IDS and generation != self.generation:
                    return
                self.hook.set_expert(self.ctx, i, calls, meta["d_model"])
                self.windows += i >= SEQ_HOOK_IDS


if __name__ == "__main__":  # compile the libraries, in a fresh process
    import warnings

    from transformer_nes.children import exit_with_parent

    exit_with_parent()
    warnings.filterwarnings("ignore", category=FutureWarning)  # torch.export's own
    for library in LIBRARIES:
        if not os.path.exists(library_path(*library)):
            compile_library(*library)
