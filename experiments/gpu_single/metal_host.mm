// Metal host for the 6502 megakernel: compiles kernel.metal at runtime,
// owns the device buffers (model tables, RAM, registers, state), and
// exposes boot/run to Python.  One run() call is one GPU dispatch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

struct Machine {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
    id<MTLBuffer> bP, bI, bF, bRam, bWr, bReg, bRegAny, bSt, bIo, bTrace,
        bDtab;
    int threads;
    int n_regs;

    Machine(const std::string& source, py::array_t<int32_t> ibuf,
            py::array_t<float> fbuf, py::array_t<int32_t> params,
            int threads_req) {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("no Metal device");
        queue = [dev newCommandQueue];

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:source.c_str()];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil
                                                 error:&err];
        if (!lib)
            throw std::runtime_error(
                std::string("metal compile: ") +
                [[err localizedDescription] UTF8String]);
        id<MTLFunction> fn = [lib newFunctionWithName:@"run6502"];
        if (!fn) throw std::runtime_error("kernel run6502 not found");
        pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso)
            throw std::runtime_error(
                std::string("pipeline: ") +
                [[err localizedDescription] UTF8String]);
        threads = threads_req;
        if ((NSUInteger)threads > pso.maxTotalThreadsPerThreadgroup)
            threads = (int)pso.maxTotalThreadsPerThreadgroup;

        auto upload = [&](const void* p, size_t n) {
            return [dev newBufferWithBytes:p length:(n ? n : 4)
                                   options:MTLResourceStorageModeShared];
        };
        bP = upload(params.data(), params.size() * 4);
        bI = upload(ibuf.data(), ibuf.size() * 4);
        bF = upload(fbuf.data(), fbuf.size() * 4);
        n_regs = params.data()[3];
        bRam = [dev newBufferWithLength:65536 * 4
                                options:MTLResourceStorageModeShared];
        bWr = [dev newBufferWithLength:65536 * 4
                               options:MTLResourceStorageModeShared];
        bReg = [dev newBufferWithLength:(n_regs ? n_regs : 1) * 3 * 4
                                options:MTLResourceStorageModeShared];
        bRegAny = [dev newBufferWithLength:(n_regs ? n_regs : 1) * 4
                                   options:MTLResourceStorageModeShared];
        bSt = [dev newBufferWithLength:13 * 4
                               options:MTLResourceStorageModeShared];
        bIo = [dev newBufferWithLength:8 * 4
                               options:MTLResourceStorageModeShared];
        // Memoized dispatch: 2 ints per pc (verified seqtab row + span
        // epoch snapshot), then 256 page write-epoch counters.
        bDtab = [dev newBufferWithLength:(2 * 65536 + 256) * 4
                                 options:MTLResourceStorageModeShared];
        bTrace = nil;
    }

    void boot(py::bytes image, int start_pc) {
        std::string img = image;
        if (img.size() > 65536) throw std::runtime_error("image > 64K");
        float* ram = (float*)bRam.contents;
        float* wr = (float*)bWr.contents;
        for (int i = 0; i < 65536; ++i) {
            unsigned char b = i < (int)img.size() ? (unsigned char)img[i] : 0;
            ram[i] = (float)b;
            wr[i] = b ? 1.0f : 0.0f;
        }
        std::memset(bReg.contents, 0, bReg.length);
        std::memset(bRegAny.contents, 0, bRegAny.length);
        std::memset(bDtab.contents, 0, bDtab.length);
        int* st = (int*)bSt.contents;
        std::memset(st, 0, 13 * 4);
        st[0] = start_pc & 0xFFFF;
        st[4] = 0xFD;
        st[5] = 0x24;
    }

    py::dict run(int max_steps, int trace) {
        if (trace == 1) {
            size_t need = (size_t)max_steps * 22 * 4;
            if (!bTrace || bTrace.length < need)
                bTrace = [dev newBufferWithLength:need
                                          options:MTLResourceStorageModeShared];
        }
        int* io = (int*)bIo.contents;
        io[0] = max_steps;
        io[1] = trace;
        io[2] = io[3] = io[4] = io[5] = io[6] = 0;

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        id<MTLBuffer> bufs[11] = {bP, bI, bF, bRam, bWr, bReg, bRegAny,
                                  bSt, bIo, bTrace ? bTrace : bIo, bDtab};
        for (int i = 0; i < 11; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            throw std::runtime_error(
                std::string("dispatch: ") +
                [[cb.error localizedDescription] UTF8String]);

        py::dict out;
        out["steps"] = io[2];
        out["halted"] = io[3];
        out["forwards"] = io[4];
        out["layers"] = io[5];
        out["lowered_forwards"] = io[6];
        if (trace == 1) {
            py::array_t<int32_t> tr({(py::ssize_t)io[2], (py::ssize_t)22});
            std::memcpy(tr.mutable_data(), bTrace.contents,
                        (size_t)io[2] * 22 * 4);
            out["trace"] = tr;
        }
        return out;
    }

    py::array_t<int32_t> trace_raw(int n) {
        // Debug window into the trace buffer (whatever the kernel last
        // wrote there, any mode).
        if (!bTrace || (size_t)n * 4 > bTrace.length)
            throw std::runtime_error("trace buffer too small");
        py::array_t<int32_t> out(n);
        std::memcpy(out.mutable_data(), bTrace.contents, (size_t)n * 4);
        return out;
    }

    py::dict state() {
        int* st = (int*)bSt.contents;
        py::dict s;
        const char* names[13] = {"pc", "a", "x", "y", "s", "p", "ir", "oplo",
                                 "ophi", "ea", "data", "halted", "nmi"};
        for (int i = 0; i < 13; ++i) s[names[i]] = st[i];
        return s;
    }
};

PYBIND11_MODULE(metal_host, m) {
    py::class_<Machine>(m, "Machine")
        .def(py::init<const std::string&, py::array_t<int32_t>,
                      py::array_t<float>, py::array_t<int32_t>, int>())
        .def("boot", &Machine::boot)
        .def("run", &Machine::run, py::arg("max_steps"),
             py::arg("trace") = false)
        .def("state", &Machine::state)
        .def("trace_raw", &Machine::trace_raw)
        .def_readonly("threads", &Machine::threads);
}
