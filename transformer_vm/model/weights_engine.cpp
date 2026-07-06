// Sparse C++ engine for analytically-constructed transformer weights.
//
// One generic translation unit (no per-model codegen): the Python side
// loads each layer's CSR matrices and head classification, and forward()
// runs the whole layer stack for one token position -- CSR matvecs for
// the projections and FFN, real hull hard-attention for lookup heads, and
// the attend-to-self shortcut for passthrough/erase heads (their keys are
// 2*pos, strictly increasing, so the argmax is provably the latest token;
// the shortcut returns exactly what the attention would).
//
// Numerics are identical to the Python sparse path: float64 throughout,
// and every FFN/residual value is a dyadic rational, so sums are exact
// under any association.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include "hull2d_cht.h"

namespace py = pybind11;

using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

struct Csr {
    std::vector<int> indptr, indices;
    std::vector<double> data;
    std::vector<int> single_rows, multi_rows;
    int rows = 0, cols = 0;

    void load(const IArr& p, const IArr& i, const DArr& d, int r, int c) {
        indptr.assign(p.data(), p.data() + p.size());
        indices.assign(i.data(), i.data() + i.size());
        data.assign(d.data(), d.data() + d.size());
        rows = r;
        cols = c;
        single_rows.clear();
        multi_rows.clear();
        single_rows.reserve(rows);
        multi_rows.reserve(rows);
        for (int row = 0; row < rows; ++row) {
            const int nnz = indptr[row + 1] - indptr[row];
            if (nnz == 1) {
                single_rows.push_back(row);
            } else if (nnz > 1) {
                multi_rows.push_back(row);
            }
        }
    }

    // y = A x
    void mv(const double* x, double* y) const {
        std::fill(y, y + rows, 0.0);
        for (int r : single_rows) {
            const int k = indptr[r];
            y[r] = data[k] * x[indices[k]];
        }
        for (int r : multi_rows) {
            double acc = 0.0;
            for (int k = indptr[r]; k < indptr[r + 1]; ++k) acc += data[k] * x[indices[k]];
            y[r] = acc;
        }
    }

    // y += A x
    void mv_add(const double* x, double* y) const {
        for (int r : single_rows) {
            const int k = indptr[r];
            y[r] += data[k] * x[indices[k]];
        }
        for (int r : multi_rows) {
            double acc = 0.0;
            for (int k = indptr[r]; k < indptr[r + 1]; ++k) acc += data[k] * x[indices[k]];
            y[r] += acc;
        }
    }
};

struct Layer {
    Csr in_proj, out_proj, ff_in, ff_out;
    std::vector<int> head_kind;                    // 0 unused, 1 lookup, 2 passthrough
    std::vector<int> tiebreak;                     // 1 = LATEST (lookup heads)
    std::vector<HardAttentionHead> hulls;          // one per lookup head
    std::vector<int> hull_of_head;                 // head -> hull index or -1
};

struct WeightsEngine {
    int d_model, n_heads, d_ffn = 0;
    std::vector<Layer> layers;
    long seq = -1;
    std::vector<double> qkv, head_out, gv, act;

    WeightsEngine(int d, int heads) : d_model(d), n_heads(heads) {}

    void add_layer(IArr ip_p, IArr ip_i, DArr ip_d,
                   IArr op_p, IArr op_i, DArr op_d,
                   IArr fi_p, IArr fi_i, DArr fi_d,
                   IArr fo_p, IArr fo_i, DArr fo_d,
                   int ffn, IArr kinds, IArr latest) {
        layers.emplace_back();
        Layer& L = layers.back();
        L.in_proj.load(ip_p, ip_i, ip_d, 3 * d_model, d_model);
        L.out_proj.load(op_p, op_i, op_d, d_model, d_model);
        L.ff_in.load(fi_p, fi_i, fi_d, 2 * ffn, d_model);
        L.ff_out.load(fo_p, fo_i, fo_d, d_model, ffn);
        L.head_kind.assign(kinds.data(), kinds.data() + kinds.size());
        L.tiebreak.assign(latest.data(), latest.data() + latest.size());
        if ((int)L.head_kind.size() != n_heads)
            throw std::runtime_error("head_kind size != n_heads");
        L.hull_of_head.assign(n_heads, -1);
        for (int h = 0; h < n_heads; ++h) {
            if (L.head_kind[h] == 1) {
                L.hull_of_head[h] = (int)L.hulls.size();
                L.hulls.emplace_back();
            }
        }
        d_ffn = std::max(d_ffn, ffn);
        qkv.assign(3 * d_model, 0.0);
        head_out.assign(d_model, 0.0);
        gv.assign(2 * d_ffn, 0.0);
        act.assign(d_ffn, 0.0);
    }

    void reset() {
        for (auto& L : layers)
            for (auto& h : L.hulls) h.clear();
        seq = -1;
    }

    // One token position through the whole layer stack, in place.
    void forward(DArr x_arr) {
        if ((int)x_arr.size() != d_model) throw std::runtime_error("x size != d_model");
        double* x = x_arr.mutable_data();
        ++seq;
        for (auto& L : layers) {
            L.in_proj.mv(x, qkv.data());
            const double* q = qkv.data();
            const double* k = qkv.data() + d_model;
            const double* v = qkv.data() + 2 * d_model;
            for (int h = 0; h < n_heads; ++h) {
                double* out = head_out.data() + 2 * h;
                switch (L.head_kind[h]) {
                    case 1: {  // lookup: genuine hull hard-attention
                        HardAttentionHead& hull = L.hulls[L.hull_of_head[h]];
                        const double key[2] = {k[2 * h], k[2 * h + 1]};
                        const double val[2] = {v[2 * h], v[2 * h + 1]};
                        hull.insert(key, val, seq);
                        const double query[2] = {q[2 * h], q[2 * h + 1]};
                        out[0] = 0.0;
                        out[1] = 0.0;
                        hull.query(query, L.tiebreak[h] ? TieBreak::LATEST : TieBreak::AVERAGE, out);
                        break;
                    }
                    case 2:  // passthrough: argmax is provably self
                        out[0] = v[2 * h];
                        out[1] = v[2 * h + 1];
                        break;
                    default:  // unused head: zero weights, zero output
                        out[0] = 0.0;
                        out[1] = 0.0;
                }
            }
            L.out_proj.mv_add(head_out.data(), x);

            L.ff_in.mv(x, gv.data());
            const int f = L.ff_in.rows / 2;
            for (int j = 0; j < f; ++j) {
                const double g = gv[j];
                act[j] = g > 0.0 ? g * gv[f + j] : 0.0;
            }
            // fo has `f` columns for this layer; act[0..f) is what it reads.
            L.ff_out.mv_add(act.data(), x);
        }
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<WeightsEngine>(m, "WeightsEngine", py::module_local())
        .def(py::init<int, int>())
        .def("add_layer", &WeightsEngine::add_layer)
        .def("reset", &WeightsEngine::reset)
        .def("forward", &WeightsEngine::forward);
}
