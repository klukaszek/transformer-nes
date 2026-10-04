PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Context>(m, "Context", py::module_local())
        .def(py::init([](uint64_t ram, uint64_t on, uint64_t reads) {
            return std::make_unique<Context>(Context{reinterpret_cast<const uint8_t*>(ram),
                reinterpret_cast<const bool*>(on), reinterpret_cast<unsigned int*>(reads)});
        }))
        .def_property_readonly("address", [](Context& c) { return reinterpret_cast<uint64_t>(&c); })
        .def_property_readonly("counts", [](const Context& c) { return std::make_pair(c.calls, c.declined); })
        .def("evaluate", [](Context& c, int id, std::vector<int32_t> x) {
            if (id < 0 || id >= (int)std::size(circuits) || x.size() != (size_t)circuits[id].width)
                throw std::invalid_argument("invalid expert or residual width");
            forward(&c, id, x.data());
            return x;
        });
    m.def("forward_address", [] { return reinterpret_cast<uint64_t>(&forward); });
    m.attr("n_experts") = std::size(circuits) - 1;
    m.attr("union_id") = std::size(circuits) - 1;
    m.attr("n_models") = std::size(circuits);
}
