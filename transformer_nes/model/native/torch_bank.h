// Raw-buffer adapter for fused Inductor kernels. Neural arithmetic stays in Torch.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace py = pybind11;
using Kernel = void (*)(const int32_t*, const uint8_t*, int32_t*, int32_t*);
struct Circuit {
    Kernel run;
    int width, aux;
    bool checks_division;
};
static const Circuit circuits[] = {@ENTRIES@};
struct Context {
    const uint8_t* ram;
    const bool* on = nullptr;
    unsigned int* reads = nullptr;
    long calls = 0, declined = 0;
};
static void invoke(Kernel kernel, const int32_t* x, const uint8_t* ram,
                   int32_t* out, int32_t* aux) {
    auto* previous = inductor_cpu_integer_div_error_flag;
    kernel(x, ram, out, aux);
    inductor_cpu_integer_div_error_flag = previous;
}
static void invoke(const Circuit& circuit, const int32_t* x, const uint8_t* ram,
                   int32_t* out, int32_t* aux) {
    if (circuit.checks_division) invoke(circuit.run, x, ram, out, aux);
    else circuit.run(x, ram, out, aux);
}
static bool forward(void* context, int id, int* x) {
    auto& c = *static_cast<Context*>(context);
    if (id < 0 || id >= (int)std::size(circuits)) { ++c.declined; return false; }
    const auto& p = circuits[id];
    int32_t out[@MAX_WIDTH@], aux[@MAX_AUX@];
    invoke(p, x, c.ram, out, aux);
    std::memcpy(x, out, p.width * sizeof(int32_t));
    ++c.calls;
    if (c.on && *c.on)
        for (int i = 0; i < p.aux; ++i)
            if (aux[i] >= 0 && std::find(aux, aux + i, aux[i]) == aux + i)
                ++c.reads[aux[i] & 65535];
    return true;
}
