// Kernels write packed readouts directly into the caller's output buffer.
static constexpr int input_widths[] = {@INPUT_WIDTHS@};
struct Context {
    const uint8_t* ram;
    const bool* on = nullptr;
    unsigned int* reads = nullptr;
    long calls = 0, declined = 0;
};
static bool forward(void* context, int id, const int* x, int* out) {
    auto& c = *static_cast<Context*>(context);
    if (id < 0 || id >= (int)std::size(circuits)) { ++c.declined; return false; }
    const auto& p = circuits[id];
    int32_t aux[@MAX_AUX@];
    invoke(p, x, c.ram, out, aux);
    ++c.calls;
    if (c.on && *c.on)
        for (int i = 0; i < p.aux; ++i)
            if (aux[i] >= 0 && std::find(aux, aux + i, aux[i]) == aux + i)
                ++c.reads[aux[i] & 65535];
    return true;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Context>(m, "Context", py::module_local())
        .def(py::init([](uint64_t ram, uint64_t on, uint64_t reads) {
            return std::make_unique<Context>(Context{reinterpret_cast<const uint8_t*>(ram),
                reinterpret_cast<const bool*>(on), reinterpret_cast<unsigned int*>(reads)});
        }))
        .def_property_readonly("address", [](Context& c) { return reinterpret_cast<uint64_t>(&c); })
        .def_property_readonly("counts", [](const Context& c) { return std::make_pair(c.calls, c.declined); })
        .def("evaluate", [](Context& c, int id, std::vector<int32_t> x) {
            if (id < 0 || id >= (int)std::size(circuits) || x.size() != (size_t)input_widths[id])
                throw std::invalid_argument("invalid expert or residual width");
            std::vector<int32_t> out(circuits[id].width);
            forward(&c, id, x.data(), out.data());
            return out;
        });
    m.def("forward_address", [] { return reinterpret_cast<uint64_t>(&forward); });
    m.attr("n_experts") = std::size(circuits) - 1;
    m.attr("union_id") = std::size(circuits) - 1;
    m.attr("n_models") = std::size(circuits);
}
