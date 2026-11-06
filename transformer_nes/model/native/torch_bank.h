// Raw-buffer adapter for native kernels derived from transformer weights.
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
using Scalar = @SCALAR@;
using Kernel = void (*)(const Scalar*, const uint8_t*, Scalar*, int32_t*);
struct Circuit {
    Kernel run;
    int width, aux;
    bool checks_division;
};
static const Circuit circuits[] = {@ENTRIES@};
static void invoke(Kernel kernel, const Scalar* x, const uint8_t* ram,
                   Scalar* out, int32_t* aux) {
    kernel(x, ram, out, aux);
}
static void invoke(const Circuit& circuit, const Scalar* x, const uint8_t* ram,
                   Scalar* out, int32_t* aux) {
    if (circuit.checks_division) invoke(circuit.run, x, ram, out, aux);
    else circuit.run(x, ram, out, aux);
}
