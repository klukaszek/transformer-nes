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
    for (int i = 0; i < bank.n_circuits; ++i)
        if (bank.circuits[i].width > kMaxValues || bank.tokens[i].n_in + 3 > kMaxValues ||
            bank.tokens[i].n_out > kMaxValues)
            throw std::invalid_argument("token wider than the runtime holds");
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
    bool fixed_point;
    int ablate_slot = 0, ablate_row = -1;
    Scalar ablate_value = 0;
    py::object debug_loader;
    Kernel<Scalar> trace_kernel = nullptr, ablate_kernel = nullptr;
    int trace_aux = 0, ablate_aux = 0;

    explicit GraphEngine(const Bank& bank_in)
        : bank(bank_in), ram(static_cast<size_t>(bank_in.memory)), circuits(bank_in.circuits),
          tokens(bank_in.tokens), memory(ram.data()), n_models(bank_in.n_circuits),
          fixed_point(std::is_integral_v<Scalar> && bank_in.f_bits != 0) {
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
    void decode(const Scalar* result, Scalar* out, int n) const {
        if constexpr (std::is_integral_v<Scalar>) {
            const Scalar mask = (Scalar(1) << bank.f_bits) - 1;
            for (int i = 0; i < n; ++i) {
                if (result[i] & mask) throw std::runtime_error("fractional readout value");
                out[i] = result[i] >> bank.f_bits;
            }
        }
    }
    void step(int model, const Scalar* in, Scalar* out) {
        if ((unsigned)model >= (unsigned)n_models) throw std::out_of_range("no such model");
        if (model == 0 && ablate_row >= 0) return step_ablated(in, out);
        int32_t aux[kMaxAux];
        if (!fixed_point) return kernel_of<Scalar>(circuits[model])(in, memory, out, aux);
        Scalar result[kMaxValues];
        kernel_of<Scalar>(circuits[model])(in, memory, result, aux);
        decode(result, out, tokens[model].n_out);
    }
    __attribute__((noinline)) void step_ablated(const Scalar* in, Scalar* out) {
        Scalar x[kMaxValues], result[kMaxValues];
        debug_input(in, x);
        const auto kernel = debug(false);
        std::vector<int32_t> scratch(std::max(1, ablate_aux));
        kernel(x, memory, fixed_point ? result : out, scratch.data());
        if (fixed_point) decode(result, out, tokens[0].n_out);
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
        .def("ablate", [](E& e, int slot, int row, Scalar value) {
            if (row >= 0) e.debug(false);
            e.ablate_slot = slot; e.ablate_row = row; e.ablate_value = value;
        }, py::arg("slot") = 0, py::arg("row") = -1, py::arg("value") = 0)
        .def("abi", [](E& e) {
            return py::make_tuple(reinterpret_cast<std::uintptr_t>(&e),
                reinterpret_cast<std::uintptr_t>(&E::abi_write),
                reinterpret_cast<std::uintptr_t>(&E::abi_step),
                reinterpret_cast<std::uintptr_t>(&E::abi_trace),
                e.bank.trace_rows, e.bank.primary_width);
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
                        py::arg("memory") = b.memory, py::arg("f_bits") = b.f_bits);
    });
    // A new engine, with its own zeroed memory, for the named chip.
    m.def("engine", [](const std::string& name) -> py::object {
        const Bank& bank = chip_bank(name);
        if (bank.scalar == transformer_nes_bank::Float32)
            return py::cast(new GraphEngine<float>(bank), py::return_value_policy::take_ownership);
        return py::cast(new GraphEngine<int32_t>(bank), py::return_value_policy::take_ownership);
    });
}
