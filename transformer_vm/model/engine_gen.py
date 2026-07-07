"""Generated straight-line C++ engine for the analytic transformer weights.

The generic engine (weights_engine.cpp) spends its forward almost entirely
on the dependent scalar CSR gather -- loading indices[k]/data[k] and chasing
them (measured ~1.5-2 ns/nnz; the hulls are only ~8%).  This module applies
the kernel.py trick to the weights: it emits one C++ translation unit
specific to the built model, with column indices as immediates, +-1
coefficients as add/sub, and every coefficient as a float32 hexfloat
literal.

The emission replicates the generic engine's arithmetic exactly: terms in
CSR column order accumulated left to right, the same load-time fusion
(passthrough attention heads and linear FFN neurons as direct residual
maps, applied in the same order around the residual updates), the same
hull insert/query sequence, and the same direct-WRITE key construction --
with LATEST_ALPHA/KEY_OFFSET imported from graph.core instead of
hand-copied, so the generated engine cannot silently desync from the graph.

float32 only; the float64 diagnostic path stays on the generic engine.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import scipy.sparse as sp

from transformer_vm.graph.core import KEY_OFFSET, LATEST_ALPHA

_ATTENTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention"
)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_engine_cache")


def _lit(value: float) -> str:
    """A float32 value as an exact C++ hexfloat literal."""
    f = float(np.float32(value))
    if f == int(f) and abs(f) < 2**24:
        return f"{int(f)}.0f"
    return f"{f.hex()}f"


def _expr(indices, data, ref) -> str:
    """One CSR row as a left-to-right accumulation expression.

    Matches the generic engine's `acc += data[k] * x[indices[k]]` order:
    C++ `+`/`-` are left-associative, so a single chained expression
    rounds identically.
    """
    terms = []
    for idx, c in zip(indices, data, strict=True):
        c = np.float32(c)
        if c == 1.0:
            terms.append(("+", ref(idx)))
        elif c == -1.0:
            terms.append(("-", ref(idx)))
        else:
            terms.append(("+", f"{_lit(c)}*{ref(idx)}"))
    if not terms:
        return "0.0f"
    sign, first = terms[0]
    out = f"-{first}" if sign == "-" else first
    for sign, term in terms[1:]:
        out += f" {sign} {term}"
    return out


def _row(csr, r):
    lo, hi = csr.indptr[r], csr.indptr[r + 1]
    return csr.indices[lo:hi], csr.data[lo:hi]


def _csr(tensor) -> sp.csr_matrix:
    m = sp.csr_matrix(tensor.numpy())
    m.sort_indices()
    return m


def _zero_columns(m: sp.csr_matrix, cols) -> sp.csr_matrix:
    out = m.tolil()
    for c in cols:
        out[:, c] = 0
    out = out.tocsr()
    out.eliminate_zeros()
    out.sort_indices()
    return out


def _compose(outer: sp.csr_matrix, cols, inner_rows: sp.csr_matrix) -> sp.csr_matrix:
    """outer[:, cols] @ inner_rows, like the engine's direct-map fusion.

    Duplicate contributions merge in float64 and exact zeros are dropped,
    matching csr_from_triplets.
    """
    direct = (outer.tocsc()[:, list(cols)] @ inner_rows).tocsr()
    direct.eliminate_zeros()
    direct.sort_indices()
    return direct


def generate_source(model) -> tuple[str, int]:
    if KEY_OFFSET != 0:
        raise RuntimeError(
            "generated engine assumes KEY_OFFSET == 0 in the lookup key "
            "construction; regenerate the emission for a shifted key"
        )
    from transformer_vm.model.engine import _check_ffn_kinds

    d_model = model.tok.weight.shape[1]
    n_heads = model.attn[0].num_heads
    n_layers = len(model.attn)

    lines: list[str] = []
    emit = lines.append

    body: list[str] = []  # per-layer member functions
    hull_tiebreak: list[int] = []  # flat hull id -> latest?
    mem_hulls: list[list[int]] = [[] for _ in range(16)]  # group -> flat hull ids
    n_memory_heads = 0

    for li in range(n_layers):
        _check_ffn_kinds(model, li)
        ip = _csr(model.attn[li].in_proj_weight)
        op = _csr(model.attn[li].out_proj.weight)
        fi = _csr(model.ff_in[li].weight)
        fo = _csr(model.ff_out[li].weight)
        ffn = fo.shape[1]
        kinds = model.head_kinds[li]
        ffn_kinds = model.ffn_kinds[li]
        tiebreak = model.head_tiebreak[li]
        groups = model.head_cache_groups[li]
        mem_groups = model.head_memory_groups[li]

        lookup = [h for h in range(n_heads) if kinds[h] == 1]
        passthrough = [h for h in range(n_heads) if kinds[h] == 2]

        # Shared hulls, first-seen in ascending head order (engine order).
        hull_base = len(hull_tiebreak)
        hull_of_group: dict[int, int] = {}
        hull_of_head: dict[int, int] = {}
        for h in lookup:
            g = groups[h]
            if g < 0:
                raise RuntimeError(f"L{li} head {h}: negative cache group")
            if g not in hull_of_group:
                hull_of_group[g] = hull_base + len(hull_of_group)
                hull_tiebreak.append(int(tiebreak[h]))
            hull_of_head[h] = hull_of_group[g]
            if mem_groups[h] >= 0:
                mem_hulls[mem_groups[h]].append(hull_of_head[h])
                n_memory_heads += 1

        # Fusion, mirroring weights_engine.cpp add_layer.
        pt_cols = [2 * h + c for h in passthrough for c in range(2)]
        attn_direct = _compose(op, pt_cols, ip[[2 * d_model + c for c in pt_cols], :])
        op_kept = _zero_columns(op, pt_cols)

        lookup_cols = {2 * h + c for h in lookup for c in range(2)}
        for c in np.unique(op_kept.indices):
            if int(c) not in lookup_cols:
                raise RuntimeError(
                    f"L{li}: out_proj column {int(c)} survives at a "
                    "non-lookup head slot"
                )

        pt_j = [j for j in range(ffn) if ffn_kinds[j] == 2]
        ffn_direct = _compose(fo, pt_j, fi[[ffn + j for j in pt_j], :])
        fo_kept = _zero_columns(fo, pt_j)
        live = sorted(int(j) for j in np.unique(fo_kept.indices))

        b: list[str] = []
        b.append(f"    void layer_{li}(float* x) {{")

        # ── Attention: hull heads, then residual updates ─────────────
        for h in lookup:
            hid = hull_of_head[h]
            tb = "TieBreak::LATEST" if tiebreak[h] else "TieBreak::AVERAGE"
            q0 = _expr(*_row(ip, 2 * h), lambda i: f"x[{i}]")
            q1 = _expr(*_row(ip, 2 * h + 1), lambda i: f"x[{i}]")
            k0 = _expr(*_row(ip, d_model + 2 * h), lambda i: f"x[{i}]")
            k1 = _expr(*_row(ip, d_model + 2 * h + 1), lambda i: f"x[{i}]")
            v0 = _expr(*_row(ip, 2 * d_model + 2 * h), lambda i: f"x[{i}]")
            v1 = _expr(*_row(ip, 2 * d_model + 2 * h + 1), lambda i: f"x[{i}]")
            b.append(f"        float ho{2 * h}, ho{2 * h + 1};")
            b.append("        {")
            b.append(f"            if (hull_seen[{hid}] != seq) {{")
            b.append(f"                const float k0 = {k0}, k1 = {k1};")
            b.append(f"                const float v0 = {v0}, v1 = {v1};")
            b.append("                const double key[2] = {(double)k0, (double)k1};")
            b.append("                const double val[2] = {(double)v0, (double)v1};")
            b.append(f"                hulls[{hid}].insert(key, val, seq);")
            b.append(f"                hull_seen[{hid}] = seq;")
            b.append("            }")
            b.append(f"            const float q0 = {q0}, q1 = {q1};")
            b.append("            const double query[2] = {(double)q0, (double)q1};")
            b.append("            double od[2] = {0.0, 0.0};")
            b.append(f"            hulls[{hid}].query(query, {tb}, od);")
            b.append(f"            ho{2 * h} = (float)od[0];")
            b.append(f"            ho{2 * h + 1} = (float)od[1];")
            b.append("        }")
            b.append(f"        (void)ho{2 * h}; (void)ho{2 * h + 1};")

        # attn_direct reads pre-attention x; capture before updating.
        ad_rows = [r for r in range(d_model)
                   if attn_direct.indptr[r + 1] > attn_direct.indptr[r]]
        for r in ad_rows:
            b.append(f"        const float ad{r} = "
                     f"{_expr(*_row(attn_direct, r), lambda i: f'x[{i}]')};")
        for r in range(d_model):
            if op_kept.indptr[r + 1] > op_kept.indptr[r]:
                b.append(f"        x[{r}] += "
                         f"{_expr(*_row(op_kept, r), lambda i: f'ho{i}')};")
        for r in ad_rows:
            b.append(f"        x[{r}] += ad{r};")

        # ── FFN: direct map and activations read post-attention x ────
        fd_rows = [r for r in range(d_model)
                   if ffn_direct.indptr[r + 1] > ffn_direct.indptr[r]]
        for r in fd_rows:
            b.append(f"        const float fd{r} = "
                     f"{_expr(*_row(ffn_direct, r), lambda i: f'x[{i}]')};")
        for j in live:
            g = _expr(*_row(fi, j), lambda i: f"x[{i}]")
            v = _expr(*_row(fi, ffn + j), lambda i: f"x[{i}]")
            b.append(f"        float a{j};")
            b.append(f"        {{ const float g = {g}; const float v = {v}; "
                     f"a{j} = g > 0.0f ? g*v : 0.0f; }}")
        for r in range(d_model):
            if fo_kept.indptr[r + 1] > fo_kept.indptr[r]:
                b.append(f"        x[{r}] += "
                         f"{_expr(*_row(fo_kept, r), lambda i: f'a{i}')};")
        for r in fd_rows:
            b.append(f"        x[{r}] += fd{r};")

        b.append("    }")
        body.extend(b)

    n_hulls = len(hull_tiebreak)

    emit("// Generated straight-line fp32 engine (model/engine_gen.py).")
    emit("// Arithmetic replicates weights_engine.cpp exactly: CSR column")
    emit("// order, left-to-right accumulation, float32 hexfloat literals.")
    emit("#include <pybind11/pybind11.h>")
    emit("#include <pybind11/numpy.h>")
    emit("#include <cmath>")
    emit("#include <stdexcept>")
    emit('#include "hull2d_cht.h"')
    emit("")
    emit("namespace py = pybind11;")
    emit("using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;")
    emit("")
    emit("struct ENGINE_T {")
    emit(f"    static constexpr int D = {d_model};")
    emit("    long seq = -1;")
    emit(f"    HardAttentionHead hulls[{max(n_hulls, 1)}];")
    emit(f"    long hull_seen[{max(n_hulls, 1)}];")
    emit("")
    emit("    ENGINE_T() { reset(); }")
    emit("")
    emit("    void reset() {")
    emit(f"        for (int i = 0; i < {n_hulls}; ++i) {{")
    emit("            hulls[i].clear();")
    emit("            hull_seen[i] = -1;")
    emit("        }")
    emit("        seq = -1;")
    emit("    }")
    emit("")
    lines.extend(body)
    emit("")
    emit("    void forward_f32(FArr x_arr) {")
    emit('        if ((int)x_arr.size() != D) throw std::runtime_error("x size != d_model");')
    emit("        float* x = x_arr.mutable_data();")
    emit("        ++seq;")
    for li in range(n_layers):
        emit(f"        layer_{li}(x);")
    emit("    }")
    emit("")
    emit("    void insert_write_f32(int addr, int value, long pos) {")
    emit("        ++seq;")
    emit("        if (seq != pos)")
    emit('            throw std::runtime_error("direct WRITE position does not match engine sequence");')
    emit("        const int group = (addr >> 12) & 0xF;")
    emit("        const int local_i = addr - group * 4096;")
    emit("        const float local = (float)local_i;")
    emit("        const float kx_f = 2.0f * local;")
    emit("        const float inv_log_pos = (float)(1.0 / std::log(2.0) - 1.0 / std::log((double)pos + 2.0));")
    emit(f"        const float ky_f = -local * local + {_lit(LATEST_ALPHA)} * inv_log_pos;")
    emit("        const double key[2] = {(double)kx_f, (double)ky_f};")
    emit("        const double val[2] = {(double)(value & 0xFF), (double)(addr & 0xFFFF)};")
    offs = [0]
    flat: list[int] = []
    for g in range(16):
        flat.extend(mem_hulls[g])
        offs.append(len(flat))
    emit(f"        static const int offs[17] = {{{', '.join(map(str, offs))}}};")
    if flat:
        emit(f"        static const int ids[{len(flat)}] = {{{', '.join(map(str, flat))}}};")
        emit("        for (int i = offs[group]; i < offs[group + 1]; ++i) {")
        emit("            const int hid = ids[i];")
        emit("            if (hull_seen[hid] == seq) continue;")
        emit("            hulls[hid].insert(key, val, seq);")
        emit("            hull_seen[hid] = seq;")
        emit("        }")
    else:
        emit("        (void)offs; (void)key; (void)val;")
    emit("    }")
    emit("};")
    emit("")
    emit("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    emit('    py::class_<ENGINE_T>(m, "Engine", py::module_local())')
    emit("        .def(py::init<>())")
    emit("        .def(\"reset\", &ENGINE_T::reset)")
    emit("        .def(\"forward_f32\", &ENGINE_T::forward_f32)")
    emit("        .def(\"insert_write_f32\", &ENGINE_T::insert_write_f32);")
    emit("}")

    source = "\n".join(lines) + "\n"
    return source, n_memory_heads


class GeneratedWeightsEngine:
    """Drop-in for SparseWeightsEngine, float32 only, codegen inside."""

    def __init__(self, model, *, dtype: str = "float32"):
        if dtype != "float32":
            raise ValueError("the generated engine is float32-only; use "
                             "matmul='engine' for the float64 diagnostic path")
        self.dtype = dtype
        source, self.n_memory_heads = generate_source(model)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        source = source.replace("ENGINE_T", f"Engine_{digest}")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        src_path = os.path.join(_CACHE_DIR, f"engine_{digest}.cpp")
        if not os.path.exists(src_path):
            with open(src_path, "w") as f:
                f.write(source)

        from torch.utils.cpp_extension import load

        build_start = time.perf_counter()
        module = load(
            name=f"weights_gen_{digest}",
            sources=[src_path],
            extra_cflags=["-O2", "-std=c++17"],
            extra_include_paths=[_ATTENTION_DIR],
            verbose=False,
        )
        self.compile_seconds = time.perf_counter() - build_start
        self._engine = module.Engine()

    def reset(self):
        self._engine.reset()

    def forward(self, x: np.ndarray) -> None:
        self._engine.forward_f32(x)

    @property
    def supports_direct_writes(self) -> bool:
        return self.n_memory_heads > 0

    def insert_write(self, addr: int, value: int, pos: int) -> None:
        if self.n_memory_heads == 0:
            raise RuntimeError(
                "model has no memory-tagged lookup heads; a direct WRITE would "
                "be silently lost -- use a full WRITE-token forward instead"
            )
        self._engine.insert_write_f32(addr, value, pos)
