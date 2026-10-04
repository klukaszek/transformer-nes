@TOKEN_TABLE@
struct ENGINE_T {
    static constexpr int MEMORY = @MEMORY@;
    std::array<uint8_t, MEMORY> ram{};
    int ablate_slot = 0, ablate_row = -1, ablate_value = 0;
    py::object debug_loader;
    Kernel trace_kernel = nullptr, ablate_kernel = nullptr;
    int trace_aux = 0, ablate_aux = 0;

    void write(int addr, int value) {
        if ((unsigned)addr >= MEMORY) throw std::out_of_range("write outside the engine memory");
        ram[addr] = (uint8_t)value;
    }
    void debug_input(const int* in, int32_t* x) const {
        std::copy_n(in, tokens[0].n_in, x);
        x[tokens[0].n_in] = ablate_slot;
        x[tokens[0].n_in + 1] = ablate_row;
        x[tokens[0].n_in + 2] = ablate_value;
    }
    Kernel debug(bool trace) {
        auto& kernel = trace ? trace_kernel : ablate_kernel;
        if (!kernel) {
            const auto result = debug_loader(trace).cast<std::pair<std::uintptr_t, int>>();
            kernel = reinterpret_cast<Kernel>(result.first);
            (trace ? trace_aux : ablate_aux) = result.second;
        }
        return kernel;
    }
    void step(int model, const int* in, int* out) {
        if (model < 0 || model >= (int)std::size(tokens)) throw std::out_of_range("no such model");
        int32_t result[@MAX_OUTPUT@], aux[@MAX_AUX@];
        int32_t* destination = @F_BITS@ == 0 ? out : result;
        if (model == 0 && ablate_row >= 0) {
            int32_t x[@MAX_INPUT@];
            debug_input(in, x);
            const auto kernel = debug(false);
            std::vector<int32_t> scratch(std::max(1, ablate_aux));
            invoke(kernel, x, ram.data(), destination, scratch.data());
        } else {
            invoke(circuits[model], in, ram.data(), destination, aux);
        }
        if constexpr (@F_BITS@ != 0)
            for (int i = 0; i < tokens[model].n_out; ++i) {
                if (result[i] & ((1 << @F_BITS@) - 1)) throw std::runtime_error("fractional readout value");
                out[i] = result[i] >> @F_BITS@;
            }
    }

    void trace(const int* in, int* out) {
        int32_t x[@MAX_INPUT@];
        debug_input(in, x);
        const auto kernel = debug(true);
        std::vector<int32_t> aux(std::max(1, trace_aux));
        invoke(kernel, x, ram.data(), out, aux.data());
    }
};
static void abi_write(void* engine, int addr, int value) {
    static_cast<ENGINE_T*>(engine)->write(addr, value);
}
static void abi_step(void* engine, int model, const int* in, int* out) {
    static_cast<ENGINE_T*>(engine)->step(model, in, out);
}
static void abi_trace(void* engine, const int* in, int* out) {
    static_cast<ENGINE_T*>(engine)->trace(in, out);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ENGINE_T>(m, "Engine", py::module_local())
        .def(py::init<>())
        .def("write", &ENGINE_T::write)
        .def("set_debug_loader", [](ENGINE_T& e, py::object loader) { e.debug_loader = loader; })
        .def("step", [](ENGINE_T& e, int model, py::array_t<int, py::array::c_style | py::array::forcecast> in) {
            if (model < 0 || model >= (int)std::size(tokens)) throw std::out_of_range("no such model");
            if (in.size() != (long)tokens[model].n_in) throw std::invalid_argument("wrong input count");
            py::array_t<int> out((size_t)tokens[model].n_out);
            e.step(model, in.data(), out.mutable_data());
            return out;
        })
        .def("ablate", [](ENGINE_T& e, int slot, int row, int value) {
            if (row >= 0) e.debug(false);
            e.ablate_slot = slot; e.ablate_row = row; e.ablate_value = value;
        }, py::arg("slot") = 0, py::arg("row") = -1, py::arg("value") = 0)
        .def("abi", [](ENGINE_T& e) {
            return py::make_tuple(reinterpret_cast<std::uintptr_t>(&e),
                reinterpret_cast<std::uintptr_t>(&abi_write),
                reinterpret_cast<std::uintptr_t>(&abi_step),
                reinterpret_cast<std::uintptr_t>(&abi_trace), @TRACE_ROWS@, @PRIMARY_WIDTH@);
        });
}
