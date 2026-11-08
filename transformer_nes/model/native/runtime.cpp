// The runtime of a compiled model: one engine per chip, over the chip's own
// memory, running its circuits.
//
// The backend (model/backend.py) compiles this file into the same Torch
// extension as every chip's kernels and the nes_chips table (bank.h). Python
// creates engines by chip name; native callers take the raw ABI from abi(),
// so the hot path is a direct call into the chip's kernel.
#include "bank.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;
using transformer_nes_bank::Bank;
using transformer_nes_bank::Circuit;

namespace {

template <class Scalar>
using Kernel = void (*)(const Scalar*, const uint8_t*, Scalar*, int32_t*);

template <class Scalar>
Kernel<Scalar> kernel_of(const Circuit& circuit) {
    return reinterpret_cast<Kernel<Scalar>>(const_cast<void*>(circuit.run));
}

// Hot-path scratch lives on the stack at fixed sizes; banks are checked
// against these bounds once, when an engine or context is created.
constexpr int kMaxValues = 256;  // token inputs (+3 ablation) or readouts
constexpr int kMaxAux = 256;     // memory reads one kernel reports

const Bank& chip_bank(const std::string& name) {
    for (int i = 0; i < nes_chip_count; ++i)
        if (name == nes_chips[i].name) {
            if (nes_chips[i].bank->abi != TRANSFORMER_NES_BANK_ABI)
                throw std::runtime_error("chip ABI mismatch; rebuild the model");
            return *nes_chips[i].bank;
        }
    throw std::invalid_argument("no such chip: " + name);
}

int max_aux(const Bank& bank) {
    int result = 1;
    for (int i = 0; i < bank.n_circuits; ++i) result = std::max(result, bank.circuits[i].aux);
    return result;
}

void check_bounds(const Bank& bank) {
    if (max_aux(bank) > kMaxAux) throw std::invalid_argument("bank reports more reads than the runtime holds");
    for (int i = 0; i < bank.n_circuits; ++i) {
        if (bank.circuits[i].width > kMaxValues || bank.tokens[i].n_in + 3 > kMaxValues ||
            bank.tokens[i].n_out > kMaxValues)
            throw std::invalid_argument("token wider than the runtime holds");
        // Kernels write straight into the caller's output buffer.
        if (bank.circuits[i].width != bank.tokens[i].n_out)
            throw std::invalid_argument("a circuit's readouts are not its token's outputs");
    }
}

// ── Chip engines ────────────────────────────────────────────────────
// Model ids run token by token, each token's full circuit before its
// specializations; model 0 is the first token's full circuit. Ablation and
// trace replace model 0 with diagnostic kernels loaded on demand through
// `debug_loader`.

template <class Scalar>
struct GraphEngine {
    const Bank& bank;
    std::vector<uint8_t> ram;
    // Hot-path copies of the descriptor.
    const Circuit* circuits;
    const transformer_nes_bank::Token* tokens;
    uint8_t* memory;
    int n_models;
    // Reads per address, for kernels that report them (observing builds).
    std::vector<uint32_t> read_counts;
    int ablate_slot = 0, ablate_row = -1;
    Scalar ablate_value = 0;
    py::object debug_loader;
    Kernel<Scalar> trace_kernel = nullptr, ablate_kernel = nullptr;
    int trace_aux = 0, ablate_aux = 0;

    explicit GraphEngine(const Bank& bank_in)
        : bank(bank_in), ram(static_cast<size_t>(bank_in.memory)), circuits(bank_in.circuits),
          tokens(bank_in.tokens), memory(ram.data()), n_models(bank_in.n_circuits) {
        if (n_models < 1) throw std::invalid_argument("a chip needs a circuit");
        check_bounds(bank);
    }

    void write(int addr, int value) {
        if ((unsigned)addr >= ram.size()) throw std::out_of_range("write outside the engine memory");
        ram[addr] = (uint8_t)value;
    }
    void debug_input(const Scalar* in, Scalar* x) const {
        const int n = bank.tokens[0].n_in;
        std::copy_n(in, n, x);
        x[n] = ablate_slot;
        x[n + 1] = ablate_row;
        x[n + 2] = ablate_value;
    }
    Kernel<Scalar> debug(bool trace) {
        auto& kernel = trace ? trace_kernel : ablate_kernel;
        if (!kernel) {
            const auto result = debug_loader(trace).template cast<std::pair<std::uintptr_t, int>>();
            kernel = reinterpret_cast<Kernel<Scalar>>(result.first);
            (trace ? trace_aux : ablate_aux) = result.second;
        }
        return kernel;
    }
    // A fixed-point circuit's readouts in whole units, in place.
    static void decode(Scalar* out, int n, int f_bits) {
        if constexpr (std::is_integral_v<Scalar>) {
            // Branch-free: shift every readout, then check their fractions once.
            Scalar fractions = 0;
            for (int i = 0; i < n; ++i) {
                fractions |= out[i];
                out[i] >>= f_bits;
            }
            if (fractions & ((Scalar(1) << f_bits) - 1))
                throw std::runtime_error("fractional readout value");
        }
    }
    void step(int model, const Scalar* in, Scalar* out) {
        if ((unsigned)model >= (unsigned)n_models) throw std::out_of_range("no such model");
        if (model == 0 && ablate_row >= 0) return step_ablated(in, out);
        int32_t aux[kMaxAux];
        const Circuit& circuit = circuits[model];
        kernel_of<Scalar>(circuit)(in, memory, out, aux);
        if (circuit.aux) count_reads(aux, circuit.aux);
        if (circuit.f_bits) decode(out, tokens[model].n_out, circuit.f_bits);
    }
    // A reported read is its address plus its layer times the memory size,
    // or -1 when it fell outside memory.
    __attribute__((noinline)) void count_reads(const int32_t* reads, int n) {
        if (read_counts.empty()) read_counts.assign(ram.size(), 0);
        for (int i = 0; i < n; ++i)
            if (reads[i] >= 0) ++read_counts[(size_t)reads[i] % ram.size()];
    }
    __attribute__((noinline)) void step_ablated(const Scalar* in, Scalar* out) {
        Scalar x[kMaxValues];
        debug_input(in, x);
        const auto kernel = debug(false);
        std::vector<int32_t> scratch(std::max(1, ablate_aux));
        kernel(x, memory, out, scratch.data());
        if (circuits[0].f_bits) decode(out, tokens[0].n_out, circuits[0].f_bits);
    }
    void trace(const Scalar* in, Scalar* out) {
        Scalar input[kMaxValues];
        debug_input(in, input);
        const auto kernel = debug(true);
        std::vector<int32_t> aux(std::max(1, trace_aux));
        kernel(input, memory, out, aux.data());
    }

    static void abi_write(void* engine, int addr, int value) {
        static_cast<GraphEngine*>(engine)->write(addr, value);
    }
    static void abi_step(void* engine, int model, const Scalar* in, Scalar* out) {
        static_cast<GraphEngine*>(engine)->step(model, in, out);
    }
    static void abi_trace(void* engine, const Scalar* in, Scalar* out) {
        static_cast<GraphEngine*>(engine)->trace(in, out);
    }
};

template <class Scalar>
void bind_engine(py::module_& m, const char* name) {
    using E = GraphEngine<Scalar>;
    using Array = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
    py::class_<E>(m, name, py::module_local())
        .def("write", &E::write)
        .def("set_debug_loader", [](E& e, py::object loader) { e.debug_loader = std::move(loader); })
        .def("step", [](E& e, int model, Array in) {
            if (model < 0 || model >= e.n_models) throw std::out_of_range("no such model");
            if (in.size() != (long)e.bank.tokens[model].n_in) throw std::invalid_argument("wrong input count");
            py::array_t<Scalar> out((size_t)e.bank.tokens[model].n_out);
            e.step(model, in.data(), out.mutable_data());
            return out;
        })
        .def("trace", [](E& e, Array in) {
            // Model 0's residual stream after its embedding and each layer.
            if (in.size() != (long)e.bank.tokens[0].n_in) throw std::invalid_argument("wrong input count");
            py::array_t<Scalar> out({(py::ssize_t)e.bank.trace_rows, (py::ssize_t)e.bank.primary_width});
            e.trace(in.data(), out.mutable_data());
            return out;
        })
        .def("take_reads", [](E& e) {
            // Reads per address since the last call (observing builds).
            py::array_t<uint32_t> out((py::ssize_t)e.ram.size());
            if (e.read_counts.empty()) std::fill_n(out.mutable_data(), out.size(), 0u);
            else std::copy(e.read_counts.begin(), e.read_counts.end(), out.mutable_data());
            std::fill(e.read_counts.begin(), e.read_counts.end(), 0u);
            return out;
        })
        .def("ablate", [](E& e, int slot, int row, Scalar value) {
            if (row >= 0) e.debug(false);
            e.ablate_slot = slot; e.ablate_row = row; e.ablate_value = value;
        }, py::arg("slot") = 0, py::arg("row") = -1, py::arg("value") = 0)
        .def("abi", [](E& e) {
            return py::make_tuple(reinterpret_cast<std::uintptr_t>(&e),
                reinterpret_cast<std::uintptr_t>(&E::abi_write),
                reinterpret_cast<std::uintptr_t>(&E::abi_step),
                reinterpret_cast<std::uintptr_t>(&E::abi_trace),
                e.bank.trace_rows, e.bank.primary_width,
                reinterpret_cast<std::uintptr_t>(e.memory));
        });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("abi") = TRANSFORMER_NES_BANK_ABI;
    bind_engine<int32_t>(m, "EngineInt32");
    bind_engine<float>(m, "EngineFloat32");
    m.def("chips", [] {
        std::vector<std::string> names;
        for (int i = 0; i < nes_chip_count; ++i) names.emplace_back(nes_chips[i].name);
        return names;
    });
    m.def("info", [](const std::string& name) {
        const Bank& b = chip_bank(name);
        return py::dict(py::arg("scalar") = b.scalar, py::arg("n_circuits") = b.n_circuits,
                        py::arg("memory") = b.memory);
    });
    // A new engine, with its own zeroed memory, for the named chip.
    m.def("engine", [](const std::string& name) -> py::object {
        const Bank& bank = chip_bank(name);
        if (bank.scalar == transformer_nes_bank::Float32)
            return py::cast(new GraphEngine<float>(bank), py::return_value_policy::take_ownership);
        return py::cast(new GraphEngine<int32_t>(bank), py::return_value_policy::take_ownership);
    });
}
