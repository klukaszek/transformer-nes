// Sparse C++ engine for analytically-constructed transformer weights.
//
// One generic translation unit (no per-model codegen): the Python side
// loads each layer's CSR matrices and head classification, and forward()
// runs the whole layer stack for one token position -- CSR matvecs for
// the projections and FFN, real hull hard-attention for lookup heads, and
// load-time fusion of linear passthrough/erase paths into direct residual
// sparse maps.
//
// Default inference uses float32 CSR storage/accumulation; a float64 forward
// path remains available as a comparison/debug mode.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include "hull2d_cht.h"

namespace py = pybind11;

static constexpr double LATEST_ALPHA = 0.3;

using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

struct Triplet {
    int row;
    int col;
    double value;
};

struct Csr {
    std::vector<int> indptr, indices;
    std::vector<double> data;
    std::vector<float> data_f32;
    std::vector<int> single_rows, multi_rows;
    int rows = 0, cols = 0;

    void build_row_lists(const std::vector<char>* skip_rows = nullptr) {
        single_rows.clear();
        multi_rows.clear();
        single_rows.reserve(rows);
        multi_rows.reserve(rows);
        for (int row = 0; row < rows; ++row) {
            if (skip_rows != nullptr && (*skip_rows)[row]) continue;
            const int nnz = indptr[row + 1] - indptr[row];
            if (nnz == 1) {
                single_rows.push_back(row);
            } else if (nnz > 1) {
                multi_rows.push_back(row);
            }
        }
    }

    void load(const IArr& p, const IArr& i, const DArr& d, int r, int c) {
        std::vector<int> p_vec(p.data(), p.data() + p.size());
        std::vector<int> i_vec(i.data(), i.data() + i.size());
        std::vector<double> d_vec(d.data(), d.data() + d.size());
        load_vectors(std::move(p_vec), std::move(i_vec), std::move(d_vec), r, c);
    }

    void load_vectors(std::vector<int> p, std::vector<int> i, std::vector<double> d, int r, int c) {
        indptr = std::move(p);
        indices = std::move(i);
        data = std::move(d);
        data_f32.assign(data.begin(), data.end());
        rows = r;
        cols = c;
        build_row_lists();
    }

    void skip_rows(const std::vector<char>& skip) {
        if ((int)skip.size() != rows) throw std::runtime_error("skip row mask has wrong size");
        build_row_lists(&skip);
    }

    bool has_work() const {
        return !single_rows.empty() || !multi_rows.empty();
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

    // y = A x, float32 storage and accumulation.
    void mv_f32(const float* x, float* y) const {
        std::fill(y, y + rows, 0.0f);
        for (int r : single_rows) {
            const int k = indptr[r];
            y[r] = data_f32[k] * x[indices[k]];
        }
        for (int r : multi_rows) {
            float acc = 0.0f;
            for (int k = indptr[r]; k < indptr[r + 1]; ++k) acc += data_f32[k] * x[indices[k]];
            y[r] = acc;
        }
    }

    // y += A x, float32 storage and accumulation.
    void mv_add_f32(const float* x, float* y) const {
        for (int r : single_rows) {
            const int k = indptr[r];
            y[r] += data_f32[k] * x[indices[k]];
        }
        for (int r : multi_rows) {
            float acc = 0.0f;
            for (int k = indptr[r]; k < indptr[r + 1]; ++k) acc += data_f32[k] * x[indices[k]];
            y[r] += acc;
        }
    }
};

static Csr csr_from_triplets(int rows, int cols, std::vector<Triplet>& trips) {
    std::sort(trips.begin(), trips.end(), [](const Triplet& a, const Triplet& b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });

    std::vector<int> indptr(rows + 1, 0);
    std::vector<int> indices;
    std::vector<double> data;
    size_t p = 0;
    for (int row = 0; row < rows; ++row) {
        while (p < trips.size() && trips[p].row == row) {
            const int col = trips[p].col;
            double acc = 0.0;
            while (p < trips.size() && trips[p].row == row && trips[p].col == col) {
                acc += trips[p].value;
                ++p;
            }
            if (acc != 0.0) {
                indices.push_back(col);
                data.push_back(acc);
            }
        }
        indptr[row + 1] = (int)indices.size();
    }

    Csr out;
    out.load_vectors(std::move(indptr), std::move(indices), std::move(data), rows, cols);
    return out;
}

static Csr filter_columns_and_build_direct(Csr& out_proj, const Csr& src_proj,
                                           const std::vector<int>& col_to_src_row,
                                           int direct_rows, int direct_cols) {
    if ((int)col_to_src_row.size() != out_proj.cols)
        throw std::runtime_error("column fusion map has wrong size");

    std::vector<int> kept_indptr(out_proj.rows + 1, 0);
    std::vector<int> kept_indices;
    std::vector<double> kept_data;
    std::vector<Triplet> direct_trips;

    for (int row = 0; row < out_proj.rows; ++row) {
        for (int k = out_proj.indptr[row]; k < out_proj.indptr[row + 1]; ++k) {
            const int col = out_proj.indices[k];
            const double out_coeff = out_proj.data[k];
            const int src_row = col_to_src_row[col];
            if (src_row < 0) {
                kept_indices.push_back(col);
                kept_data.push_back(out_coeff);
                continue;
            }
            for (int sk = src_proj.indptr[src_row]; sk < src_proj.indptr[src_row + 1]; ++sk) {
                direct_trips.push_back({row, src_proj.indices[sk], out_coeff * src_proj.data[sk]});
            }
        }
        kept_indptr[row + 1] = (int)kept_indices.size();
    }

    out_proj.load_vectors(std::move(kept_indptr), std::move(kept_indices), std::move(kept_data),
                          out_proj.rows, out_proj.cols);
    return csr_from_triplets(direct_rows, direct_cols, direct_trips);
}

static std::vector<char> columns_with_nnz(const Csr& m, int n_cols) {
    std::vector<char> active(n_cols, 0);
    for (int row = 0; row < m.rows; ++row) {
        for (int k = m.indptr[row]; k < m.indptr[row + 1]; ++k) {
            active[m.indices[k]] = 1;
        }
    }
    return active;
}

static void add_vec(const double* src, double* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
}

static void add_vec_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
}

struct Layer {
    Csr in_proj, out_proj, ff_in, ff_out;
    Csr attn_direct, ffn_direct;
    std::vector<int> head_kind;                    // 0 unused, 1 lookup, 2 passthrough
    std::vector<int> ffn_kind;                     // 0 unused, 1 ReGLU, 2 linear passthrough
    std::vector<int> tiebreak;                     // 1 = LATEST (lookup heads)
    std::vector<int> lookup_heads;
    std::vector<int> passthrough_heads;
    std::vector<int> ffn_neurons;
    std::vector<std::vector<int>> memory_heads_by_group;
    std::vector<HardAttentionHead> hulls;          // shared by lookup heads with identical K/V
    std::vector<int> hull_of_head;                 // head -> hull index or -1
    std::vector<long> hull_insert_seq;             // hull index -> last token seq inserted
};

struct WeightsEngine {
    int d_model, n_heads, d_ffn = 0;
    std::vector<Layer> layers;
    long seq = -1;
    std::vector<double> qkv, head_out, gv, act, delta;
    std::vector<float> qkv_f32, head_out_f32, gv_f32, act_f32, delta_f32;

    WeightsEngine(int d, int heads) : d_model(d), n_heads(heads) {}

    void add_layer(IArr ip_p, IArr ip_i, DArr ip_d,
                   IArr op_p, IArr op_i, DArr op_d,
                   IArr fi_p, IArr fi_i, DArr fi_d,
                   IArr fo_p, IArr fo_i, DArr fo_d,
                   int ffn, IArr kinds, IArr latest, IArr ffn_kinds,
                   IArr cache_groups, IArr memory_groups) {
        layers.emplace_back();
        Layer& L = layers.back();
        L.head_kind.assign(kinds.data(), kinds.data() + kinds.size());
        L.ffn_kind.assign(ffn_kinds.data(), ffn_kinds.data() + ffn_kinds.size());
        L.tiebreak.assign(latest.data(), latest.data() + latest.size());
        std::vector<int> groups(cache_groups.data(), cache_groups.data() + cache_groups.size());
        std::vector<int> mem_groups(memory_groups.data(), memory_groups.data() + memory_groups.size());
        if ((int)L.head_kind.size() != n_heads)
            throw std::runtime_error("head_kind size != n_heads");
        if ((int)L.ffn_kind.size() != ffn)
            throw std::runtime_error("ffn_kind size != d_ffn");
        if ((int)groups.size() != n_heads)
            throw std::runtime_error("cache_group size != n_heads");
        if ((int)mem_groups.size() != n_heads)
            throw std::runtime_error("memory_group size != n_heads");

        L.in_proj.load(ip_p, ip_i, ip_d, 3 * d_model, d_model);
        L.out_proj.load(op_p, op_i, op_d, d_model, d_model);
        L.ff_in.load(fi_p, fi_i, fi_d, 2 * ffn, d_model);
        L.ff_out.load(fo_p, fo_i, fo_d, d_model, ffn);

        L.memory_heads_by_group.assign(16, {});
        L.hull_of_head.assign(n_heads, -1);
        int max_group = -1;
        for (int h = 0; h < n_heads; ++h) {
            if (L.head_kind[h] == 1) {
                L.lookup_heads.push_back(h);
                max_group = std::max(max_group, groups[h]);
                if (mem_groups[h] >= 0) {
                    if (mem_groups[h] >= 16) throw std::runtime_error("memory group out of range");
                    L.memory_heads_by_group[mem_groups[h]].push_back(h);
                }
            } else if (L.head_kind[h] == 2) {
                L.passthrough_heads.push_back(h);
            }
        }
        std::vector<int> hull_of_group(max_group + 1, -1);
        for (int h : L.lookup_heads) {
            const int group = groups[h];
            if (group < 0) throw std::runtime_error("lookup head has negative cache group");
            if (hull_of_group[group] < 0) {
                hull_of_group[group] = (int)L.hulls.size();
                L.hulls.emplace_back();
            }
            L.hull_of_head[h] = hull_of_group[group];
        }
        L.hull_insert_seq.assign(L.hulls.size(), -1);

        // Passthrough heads and linear FFN neurons are exact residual linear
        // maps.  Fuse their output columns into direct D->D CSR deltas, then
        // skip the projection rows that only existed to route those values.
        std::vector<int> attn_col_to_src_row(d_model, -1);
        std::vector<char> skip_ip_rows(3 * d_model, 0);
        for (int h : L.passthrough_heads) {
            for (int comp = 0; comp < 2; ++comp) {
                const int col = 2 * h + comp;
                skip_ip_rows[col] = 1;
                skip_ip_rows[d_model + col] = 1;
                skip_ip_rows[2 * d_model + col] = 1;
                attn_col_to_src_row[col] = 2 * d_model + col;
            }
        }
        L.attn_direct = filter_columns_and_build_direct(
            L.out_proj, L.in_proj, attn_col_to_src_row, d_model, d_model);
        L.in_proj.skip_rows(skip_ip_rows);

        std::vector<int> ffn_col_to_src_row(ffn, -1);
        for (int j = 0; j < ffn; ++j) {
            if (L.ffn_kind[j] == 2) ffn_col_to_src_row[j] = ffn + j;
        }
        L.ffn_direct = filter_columns_and_build_direct(
            L.ff_out, L.ff_in, ffn_col_to_src_row, d_model, d_model);

        std::vector<char> active_ffn = columns_with_nnz(L.ff_out, ffn);
        std::vector<char> skip_ffn_rows(2 * ffn, 0);
        for (int j = 0; j < ffn; ++j) {
            if (active_ffn[j]) {
                L.ffn_neurons.push_back(j);
            } else {
                skip_ffn_rows[j] = 1;
                skip_ffn_rows[ffn + j] = 1;
            }
        }
        L.ff_in.skip_rows(skip_ffn_rows);

        d_ffn = std::max(d_ffn, ffn);
        qkv.assign(3 * d_model, 0.0);
        head_out.assign(d_model, 0.0);
        gv.assign(2 * d_ffn, 0.0);
        act.assign(d_ffn, 0.0);
        delta.assign(d_model, 0.0);
        qkv_f32.assign(3 * d_model, 0.0f);
        head_out_f32.assign(d_model, 0.0f);
        gv_f32.assign(2 * d_ffn, 0.0f);
        act_f32.assign(d_ffn, 0.0f);
        delta_f32.assign(d_model, 0.0f);
    }

    void reset() {
        for (auto& L : layers)
            for (auto& h : L.hulls) h.clear();
        for (auto& L : layers)
            std::fill(L.hull_insert_seq.begin(), L.hull_insert_seq.end(), -1);
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
            for (int h : L.lookup_heads) {
                double* out = head_out.data() + 2 * h;
                const int hull_idx = L.hull_of_head[h];
                HardAttentionHead& hull = L.hulls[hull_idx];
                if (L.hull_insert_seq[hull_idx] != seq) {
                    const double key[2] = {k[2 * h], k[2 * h + 1]};
                    const double val[2] = {v[2 * h], v[2 * h + 1]};
                    hull.insert(key, val, seq);
                    L.hull_insert_seq[hull_idx] = seq;
                }
                const double query[2] = {q[2 * h], q[2 * h + 1]};
                out[0] = 0.0;
                out[1] = 0.0;
                hull.query(query, L.tiebreak[h] ? TieBreak::LATEST : TieBreak::AVERAGE, out);
            }
            const bool has_attn_direct = L.attn_direct.has_work();
            if (has_attn_direct) L.attn_direct.mv(x, delta.data());
            L.out_proj.mv_add(head_out.data(), x);
            if (has_attn_direct) add_vec(delta.data(), x, d_model);

            const bool has_ffn_direct = L.ffn_direct.has_work();
            if (has_ffn_direct) L.ffn_direct.mv(x, delta.data());
            L.ff_in.mv(x, gv.data());
            const int f = L.ff_in.rows / 2;
            for (int j : L.ffn_neurons) {
                const double g = gv[j];
                act[j] = g > 0.0 ? g * gv[f + j] : 0.0;
            }
            // fo has `f` columns for this layer; act[0..f) is what it reads.
            L.ff_out.mv_add(act.data(), x);
            if (has_ffn_direct) add_vec(delta.data(), x, d_model);
        }
    }

    // One token position through the whole layer stack with float32 linear maps.
    void forward_f32(FArr x_arr) {
        if ((int)x_arr.size() != d_model) throw std::runtime_error("x size != d_model");
        float* x = x_arr.mutable_data();
        ++seq;
        for (auto& L : layers) {
            L.in_proj.mv_f32(x, qkv_f32.data());
            const float* q = qkv_f32.data();
            const float* k = qkv_f32.data() + d_model;
            const float* v = qkv_f32.data() + 2 * d_model;
            for (int h : L.lookup_heads) {
                float* out = head_out_f32.data() + 2 * h;
                const int hull_idx = L.hull_of_head[h];
                HardAttentionHead& hull = L.hulls[hull_idx];
                if (L.hull_insert_seq[hull_idx] != seq) {
                    const double key[2] = {(double)k[2 * h], (double)k[2 * h + 1]};
                    const double val[2] = {(double)v[2 * h], (double)v[2 * h + 1]};
                    hull.insert(key, val, seq);
                    L.hull_insert_seq[hull_idx] = seq;
                }
                const double query[2] = {(double)q[2 * h], (double)q[2 * h + 1]};
                double out_d[2] = {0.0, 0.0};
                hull.query(query, L.tiebreak[h] ? TieBreak::LATEST : TieBreak::AVERAGE, out_d);
                out[0] = (float)out_d[0];
                out[1] = (float)out_d[1];
            }
            const bool has_attn_direct = L.attn_direct.has_work();
            if (has_attn_direct) L.attn_direct.mv_f32(x, delta_f32.data());
            L.out_proj.mv_add_f32(head_out_f32.data(), x);
            if (has_attn_direct) add_vec_f32(delta_f32.data(), x, d_model);

            const bool has_ffn_direct = L.ffn_direct.has_work();
            if (has_ffn_direct) L.ffn_direct.mv_f32(x, delta_f32.data());
            L.ff_in.mv_f32(x, gv_f32.data());
            const int f = L.ff_in.rows / 2;
            for (int j : L.ffn_neurons) {
                const float g = gv_f32[j];
                act_f32[j] = g > 0.0f ? g * gv_f32[f + j] : 0.0f;
            }
            L.ff_out.mv_add_f32(act_f32.data(), x);
            if (has_ffn_direct) add_vec_f32(delta_f32.data(), x, d_model);
        }
    }

    void insert_write_f32(int addr, int value, long pos) {
        ++seq;
        if (seq != pos) throw std::runtime_error("direct WRITE position does not match engine sequence");

        const int group = (addr >> 12) & 0xF;
        const int local_i = addr - group * 4096;
        const float local = (float)local_i;
        const float kx_f = 2.0f * local;
        const float inv_log_pos = (float)(1.0 / std::log(2.0) - 1.0 / std::log((double)pos + 2.0));
        const float ky_f = -local * local + (float)LATEST_ALPHA * inv_log_pos;
        const double key[2] = {(double)kx_f, (double)ky_f};
        const double val[2] = {(double)(value & 0xFF), (double)(addr & 0xFFFF)};

        for (auto& L : layers) {
            for (int h : L.memory_heads_by_group[group]) {
                const int hull_idx = L.hull_of_head[h];
                if (L.hull_insert_seq[hull_idx] == seq) continue;
                L.hulls[hull_idx].insert(key, val, seq);
                L.hull_insert_seq[hull_idx] = seq;
            }
        }
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<WeightsEngine>(m, "WeightsEngine", py::module_local())
        .def(py::init<int, int>())
        .def("add_layer", &WeightsEngine::add_layer)
        .def("reset", &WeightsEngine::reset)
        .def("forward", &WeightsEngine::forward)
        .def("forward_f32", &WeightsEngine::forward_f32)
        .def("insert_write_f32", &WeightsEngine::insert_write_f32);
}
