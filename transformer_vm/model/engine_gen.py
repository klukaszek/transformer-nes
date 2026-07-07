"""Generated straight-line C++ engine for the analytic transformer weights.

The generic engine (weights_engine.cpp) spends its forward almost entirely
on the dependent scalar CSR gather -- loading indices[k]/data[k] and chasing
them (measured ~1.5-2 ns/nnz; the hulls are only ~8%).  This module applies
the kernel.py trick to the weights: it emits one C++ translation unit
specific to the built model, with column indices as immediates and +-1
coefficients as add/sub.

Three arithmetic modes:

- ``arith="int32"`` (default): the residual stream is exact fixed-point
  int32 at scale 2**F (F=1 covers the half-integer values the +-0.5
  direct-map coefficients produce; every rescaling shift is guarded and
  throws rather than round).  Every residual-feeding coefficient is an
  integer, all lookup heads are LATEST tie-break so hull outputs return
  the values that were inserted, and the STATE/HALT head rows are
  integers -- the whole non-attention computation is exact integer
  arithmetic.  Partial sums may wrap mod 2^32 (-fwrapv); every final
  value fits, so results are exact.  Q/K/V and the position features
  live in double, feeding the hulls directly (hulls always resolve in
  double anyway), and clear_key-gated inserts (ky -= 1e30, provably
  never a query winner) are skipped entirely.

- ``arith="int64"``: the same emission at int64 width (no wrap argument
  needed); measured slower than int32 -- the linear part is load-bound,
  so value density wins.

- ``arith="f32"``: bit-for-bit replica of the generic engine's float32
  arithmetic (CSR column order, left-to-right accumulation, float32
  hexfloat literals); kept as the cross-check and benchmarking reference.

Both modes generate the direct-WRITE key construction from graph.core's
LATEST_ALPHA/KEY_OFFSET instead of hand-copying it, so the generated
engine cannot silently desync from the graph.  The float64 diagnostic
path stays on the generic engine.
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


def _lit_d(value: float) -> str:
    """A float64 value as an exact C++ hexfloat literal."""
    f = float(value)
    if f == int(f) and abs(f) < 2**53:
        return f"{int(f)}.0"
    return f.hex()


def _join(terms) -> str:
    if not terms:
        return ""
    sign, first = terms[0]
    out = f"-{first}" if sign == "-" else first
    for sign, term in terms[1:]:
        out += f" {sign} {term}"
    return out


def _expr(indices, data, ref) -> str:
    """One CSR row as a float32 left-to-right accumulation expression.

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
    return _join(terms) or "0.0f"


def _expr_dbl(indices, data, ref) -> str:
    """One CSR row as a float64 expression (hull-bound Q/K/V rows)."""
    terms = []
    for idx, c in zip(indices, data, strict=True):
        c = float(c)
        if c == 1.0:
            terms.append(("+", ref(idx)))
        elif c == -1.0:
            terms.append(("-", ref(idx)))
        else:
            terms.append(("+", f"{_lit_d(c)}*{ref(idx)}"))
    return _join(terms) or "0.0"


def _expr_int(indices, data, ref, where: str) -> str:
    """One CSR row as an exact int64 expression.

    Integer coefficients only, except +-0.5 (four fused direct-map
    coefficients) which become guarded halvings; anything else means the
    integer-residual invariant does not hold for this model, so refuse to
    generate rather than silently round.
    """
    terms = []
    for idx, c in zip(indices, data, strict=True):
        c = float(c)
        if c == 1.0:
            terms.append(("+", ref(idx)))
        elif c == -1.0:
            terms.append(("-", ref(idx)))
        elif c == 0.5:
            terms.append(("+", f"half({ref(idx)})"))
        elif c == -0.5:
            terms.append(("-", f"half({ref(idx)})"))
        elif c == int(c):
            sign = "+" if c > 0 else "-"
            terms.append((sign, f"{abs(int(c))}*{ref(idx)}"))
        else:
            raise RuntimeError(
                f"{where}: coefficient {c} is not representable in the "
                "exact int64 emission; use arith='f32'"
            )
    return _join(terms) or "0"


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


def generate_source(model, step_spec: dict | None = None,
                    arith: str = "int64") -> tuple[str, int, bool]:
    """Emit the C++ source; returns (source, n_memory_heads, has_step).

    step_spec, if given, additionally emits a machine_step member that runs
    one whole 6502 instruction in C++ -- STATE-token embedding, position
    features, the layer stack, readout, write extraction with direct hull
    inserts, and the STATE/HALT head logits -- writing results into a
    caller-provided int64 buffer.  Layout: {"inputs": {name: slot},
    "p_flags": [(slot, mask)], "outputs": {name: slot},
    "halt_head"/"state_head": (cols, vals)}.
    """
    if KEY_OFFSET != 0:
        raise RuntimeError(
            "generated engine assumes KEY_OFFSET == 0 in the lookup key "
            "construction; regenerate the emission for a shifted key"
        )
    if arith not in ("int64", "int32", "f32"):
        raise ValueError(f"unknown arith {arith!r}")
    is_int = arith in ("int64", "int32")
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

    # Residual references.  In int mode the position features (slots 0-2)
    # never feed integer rows -- only hull-bound Q/K/V rows read them, via
    # the pf[] doubles -- so integer refs to slots 0-2 are a hard error.
    if is_int:
        # int32 relies on mod-2^32 wrap (-fwrapv): partial sums may wrap,
        # but every final graph value fits, so results are exact.
        elem, zero = ("long long", "0") if arith == "int64" else ("int", "0")
        # Fixed-point scale: residual holds value * 2**F.  F=1 covers the
        # half-integer values the +-0.5 direct-map coefficients produce;
        # every shift is guarded, so an insufficient F fails loudly.
        f_bits = 1
        inv_scale = f"0x1p-{f_bits}"

        def xref(i):
            if i <= 2:
                raise RuntimeError(
                    "integer row reads a position slot; the int64 emission "
                    "keeps position features out of the residual"
                )
            return f"x[{i}]"

        def dref(i):
            return f"pf[{i}]" if i <= 2 else f"({inv_scale}*(double)x[{i}])"

        def row_int(csr_m, r, ref, where):
            return _expr_int(*_row(csr_m, r), ref, where)
    else:
        elem, zero = "float", "0.0f"

        def xref(i):
            return f"x[{i}]"

        def dref(i):  # unused in f32 mode
            return f"x[{i}]"

        def row_int(csr_m, r, ref, where):
            return _expr(*_row(csr_m, r), ref)

    layer_sig = f"{elem}* x, const double* pf" if is_int else "float* x"

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
        if is_int:
            for h in lookup:
                if not tiebreak[h]:
                    raise RuntimeError(
                        f"L{li} head {h}: AVERAGE tie-break can return "
                        "fractional hull outputs; use arith='f32'"
                    )

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
        b.append(f"    void layer_{li}({layer_sig}) {{")

        # ── Attention: hull heads, then residual updates ─────────────
        for h in lookup:
            hid = hull_of_head[h]
            tb = "TieBreak::LATEST" if tiebreak[h] else "TieBreak::AVERAGE"
            if is_int:
                # Q/K/V straight to double: the hull resolves in double
                # anyway, and only these rows may read position features.
                q0 = _expr_dbl(*_row(ip, 2 * h), dref)
                q1 = _expr_dbl(*_row(ip, 2 * h + 1), dref)
                k0 = _expr_dbl(*_row(ip, d_model + 2 * h), dref)
                k1 = _expr_dbl(*_row(ip, d_model + 2 * h + 1), dref)
                v0 = _expr_dbl(*_row(ip, 2 * d_model + 2 * h), dref)
                v1 = _expr_dbl(*_row(ip, 2 * d_model + 2 * h + 1), dref)
                b.append(f"        {elem} ho{2 * h}, ho{2 * h + 1};")
                b.append("        {")
                b.append(f"            if (hull_seen[{hid}] != seq) {{")
                b.append(f"                const double key[2] = {{{k0}, {k1}}};")
                b.append("                // clear_key-gated tokens (ky -= 1e30) can never")
                b.append("                // win a query; skipping the insert is invisible.")
                b.append("                if (key[1] > -1.0e29) {")
                b.append(f"                    const double val[2] = {{{v0}, {v1}}};")
                b.append(f"                    hulls[{hid}].insert(key, val, seq);")
                b.append("                }")
                b.append(f"                hull_seen[{hid}] = seq;")
                b.append("            }")
                b.append(f"            const double query[2] = {{{q0}, {q1}}};")
                b.append("            double od[2] = {0.0, 0.0};")
                b.append(f"            hulls[{hid}].query(query, {tb}, od);")
                b.append("            const double s0 = od[0] * SCALE_D, "
                         "s1 = od[1] * SCALE_D;")
                b.append("            const long long t0 = (long long)s0, "
                         "t1 = (long long)s1;")
                b.append(f"            ho{2 * h} = ({elem})t0;")
                b.append(f"            ho{2 * h + 1} = ({elem})t1;")
                b.append(f"            if ((double)t0 != s0 || (double)t1 != s1 "
                         f"|| ho{2 * h} != t0 || ho{2 * h + 1} != t1)")
                b.append('                throw std::runtime_error('
                         '"hull output not representable at the fixed-point '
                         'scale");')
                b.append("        }")
            else:
                q0 = _expr(*_row(ip, 2 * h), xref)
                q1 = _expr(*_row(ip, 2 * h + 1), xref)
                k0 = _expr(*_row(ip, d_model + 2 * h), xref)
                k1 = _expr(*_row(ip, d_model + 2 * h + 1), xref)
                v0 = _expr(*_row(ip, 2 * d_model + 2 * h), xref)
                v1 = _expr(*_row(ip, 2 * d_model + 2 * h + 1), xref)
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
            b.append(f"        const {elem} ad{r} = "
                     f"{row_int(attn_direct, r, xref, f'L{li} attn_direct')};")
        for r in range(d_model):
            if op_kept.indptr[r + 1] > op_kept.indptr[r]:
                b.append(f"        x[{r}] += "
                         f"{row_int(op_kept, r, lambda i: f'ho{i}', f'L{li} out_proj')};")
        for r in ad_rows:
            b.append(f"        x[{r}] += ad{r};")

        # ── FFN: direct map and activations read post-attention x ────
        fd_rows = [r for r in range(d_model)
                   if ffn_direct.indptr[r + 1] > ffn_direct.indptr[r]]
        for r in fd_rows:
            b.append(f"        const {elem} fd{r} = "
                     f"{row_int(ffn_direct, r, xref, f'L{li} ffn_direct')};")
        for j in live:
            g = row_int(fi, j, xref, f"L{li} ff_gate")
            v = row_int(fi, ffn + j, xref, f"L{li} ff_value")
            b.append(f"        {elem} a{j};")
            if is_int:
                b.append(f"        {{ const {elem} g = {g}; "
                         f"const {elem} v = {v}; "
                         f"a{j} = g > 0 ? shr(g*v) : 0; }}")
            else:
                b.append(f"        {{ const {elem} g = {g}; const {elem} v = {v}; "
                         f"a{j} = g > {zero} ? g*v : {zero}; }}")
        for r in range(d_model):
            if fo_kept.indptr[r + 1] > fo_kept.indptr[r]:
                b.append(f"        x[{r}] += "
                         f"{row_int(fo_kept, r, lambda i: f'a{i}', f'L{li} ff_out')};")
        for r in fd_rows:
            b.append(f"        x[{r}] += fd{r};")

        b.append("    }")
        body.extend(b)

    n_hulls = len(hull_tiebreak)

    emit(f"// Generated straight-line {arith} engine (model/engine_gen.py).")
    if is_int:
        emit("// Residual stream is exact int64 (integer addition is")
        emit("// associative, so the compiler may tree-reduce/vectorize);")
        emit("// Q/K/V and position features are double, hull-resolved.")
    else:
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
    emit("using LArr = py::array_t<long long, py::array::c_style | py::array::forcecast>;")
    emit("")
    emit("struct ENGINE_T {")
    emit(f"    static constexpr int D = {d_model};")
    emit("    long seq = -1;")
    emit(f"    HardAttentionHead hulls[{max(n_hulls, 1)}];")
    emit(f"    long hull_seen[{max(n_hulls, 1)}];")
    if is_int:
        emit(f"    static constexpr int F = {f_bits};")
        emit(f"    static constexpr {elem} SCALE = 1 << F;")
        emit("    static constexpr double SCALE_D = (double)SCALE;")
        emit(f"    {elem} xbuf[D];")
        emit("")
        emit(f"    static {elem} shr({elem} v) {{")
        emit("        // Rescale a product of two fixed-point values; the")
        emit("        // represented result must stay on the F-bit grid.")
        emit("        if (v & (SCALE - 1))")
        emit('            throw std::runtime_error("value finer than the fixed-point scale");')
        emit("        return v >> F;")
        emit("    }")
        emit("")
        emit(f"    static long long shr2({elem} v) {{")
        emit("        // Readout: trace-visible values are whole integers.")
        emit("        if (v & (SCALE - 1))")
        emit('            throw std::runtime_error("fractional readout value");')
        emit("        return (long long)(v >> F);")
        emit("    }")
        emit("")
        emit(f"    static {elem} half({elem} v) {{")
        emit("        // 0.5 coefficients in the fused direct maps: exact only")
        emit("        // on even operands, and the invariant says they always")
        emit("        // are -- fail loudly instead of rounding.")
        emit("        if (v & 1)")
        emit('            throw std::runtime_error("odd value under a 0.5 coefficient");')
        emit("        return v >> 1;")
        emit("    }")
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
    if is_int:
        emit("        // Machine-protocol adapter: exact-integer token values")
        emit("        // arrive as floats; run the int stack and convert back.")
        emit("        const double pf[3] = {(double)x[0], (double)x[1], (double)x[2]};")
        emit(f"        {elem}* xi = xbuf;")
        emit("        xi[0] = 0; xi[1] = 0; xi[2] = 0;")
        emit("        for (int i = 3; i < D; ++i)")
        emit(f"            xi[i] = ({elem})(long long)std::nearbyint((double)x[i] * SCALE_D);")
        for li in range(n_layers):
            emit(f"        layer_{li}(xi, pf);")
        emit("        for (int i = 3; i < D; ++i) x[i] = (float)((double)xi[i] / SCALE_D);")
    else:
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

    has_step = step_spec is not None and bool(flat)
    if has_step:
        ins = step_spec["inputs"]
        outs = step_spec["outputs"]

        def out_slot(name):
            return outs[name]

        emit("")
        emit("    // One whole 6502 instruction: STATE embed -> layers ->")
        emit("    // readout -> direct WRITE inserts -> halt logits.")
        if not is_int:
            emit(f"    float xbuf[{d_model}];")
            emit("")
            emit("    static long long rd(float v) {")
            emit("        // Python round() is half-to-even; nearbyint under the")
            emit("        // default FE_TONEAREST matches it (llround does not).")
            emit("        return (long long)std::nearbyint((double)v);")
            emit("    }")
        emit("")
        emit("    void machine_step(int pc, int a, int xr, int yr, int sr,")
        emit("                      int p, int ir, int oplo, int ophi,")
        emit("                      int ea, int data, int halted, int nmi,")
        emit("                      LArr out_arr) {")
        emit('        if ((int)out_arr.size() != 21) throw std::runtime_error("step out buffer must have 21 slots");')
        emit("        long long* out = out_arr.mutable_data();")
        emit(f"        {elem}* x = xbuf;")
        emit(f"        for (int i = 0; i < {d_model}; ++i) x[i] = {zero};")
        # Embedded values live on the fixed-point grid in int mode.
        if is_int:
            one, zero_l = "SCALE", "0"
            def sc(v, e=elem):
                return f"(({e})({v}) << F)"
        else:
            one, zero_l = "1.0f", "0.0f"
            def sc(v):
                return f"(float)({v})"
        emit(f"        x[{model.one_slot}] = {one};")
        emit(f"        x[{ins['is_state']}] = {one};")
        emit(f"        x[{ins['pc_lo']}] = {sc('pc & 0xFF')};")
        emit(f"        x[{ins['pc_hi']}] = {sc('(pc >> 8) & 0xFF')};")
        emit(f"        x[{ins['a']}] = {sc('a')};")
        emit(f"        x[{ins['x']}] = {sc('xr')};")
        emit(f"        x[{ins['y']}] = {sc('yr')};")
        emit(f"        x[{ins['s']}] = {sc('sr')};")
        for slot, mask in step_spec["p_flags"]:
            emit(f"        x[{slot}] = (p & {mask}) ? {one} : {zero_l};")
        emit(f"        x[{ins['ir']}] = {sc('ir')};")
        emit(f"        x[{ins['oplo']}] = {sc('oplo')};")
        emit(f"        x[{ins['ophi']}] = {sc('ophi')};")
        emit(f"        x[{ins['ea_lo']}] = {sc('ea & 0xFF')};")
        emit(f"        x[{ins['ea_hi']}] = {sc('(ea >> 8) & 0xFF')};")
        emit(f"        x[{ins['data']}] = {sc('data')};")
        emit(f"        x[{ins['halted']}] = {sc('halted')};")
        emit(f"        x[{ins['nmi']}] = {sc('nmi')};")
        emit("        const long long pos = seq + 1;")
        if is_int:
            emit("        const double pf[3] = {")
            emit("            (double)pos,")
            emit("            1.0 / std::log(2.0) - 1.0 / std::log((double)pos + 2.0),")
            emit("            (double)pos * (double)pos,")
            emit("        };")
        else:
            emit("        x[0] += (float)pos;")
            emit("        x[1] += (float)(1.0 / std::log(2.0) - 1.0 / std::log((double)pos + 2.0));")
            emit("        // integer square converted once, like NumPy's int -> float32")
            emit("        x[2] += (float)((unsigned long long)pos * (unsigned long long)pos);")
        emit("        ++seq;")
        for li in range(n_layers):
            emit(f"        layer_{li}(x{', pf' if is_int else ''});")
        rdv = (lambda s: f"shr2(x[{s}])") if is_int else (lambda s: f"rd(x[{s}])")
        for i, (name, mask) in enumerate((
            ("current_pc", 0xFFFF), ("new_a", 0xFF), ("new_x", 0xFF),
            ("new_y", 0xFF), ("new_s", 0xFF), ("new_p", 0xFF),
            ("fetched_opcode", 0xFF), ("fetched_operand_lo", 0xFF),
            ("fetched_operand_hi", 0xFF), ("ea", 0xFFFF), ("data_out", 0xFF),
            ("new_halted", 0xFF), ("next_pc", 0xFFFF),
        )):
            emit(f"        out[{i}] = {rdv(out_slot(name))} & 0x{mask:X};")
        emit("        long long n_writes = 0;")
        for store, addr, value in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            emit(f"        if (({rdv(out_slot(store))} & 0xFF) != 0) {{")
            emit(f"            out[15 + 2 * n_writes] = {rdv(out_slot(addr))} & 0xFFFF;")
            emit(f"            out[16 + 2 * n_writes] = {rdv(out_slot(value))} & 0xFF;")
            emit("            ++n_writes;")
            emit("        }")
        emit("        out[14] = n_writes;")
        emit("        for (long long i = 0; i < n_writes; ++i)")
        emit("            insert_write_f32((int)out[15 + 2 * i], (int)out[16 + 2 * i], seq + 1);")
        hc, hv = step_spec["halt_head"]
        sc, sv = step_spec["state_head"]
        if is_int:
            emit(f"        const long long halt_logit = "
                 f"{_expr_int(hc, hv, lambda i: f'x[{i}]', 'halt head')};")
            emit(f"        const long long state_logit = "
                 f"{_expr_int(sc, sv, lambda i: f'x[{i}]', 'state head')};")
        else:
            emit(f"        const float halt_logit = {_expr(hc, hv, lambda i: f'x[{i}]')};")
            emit(f"        const float state_logit = {_expr(sc, sv, lambda i: f'x[{i}]')};")
        emit("        out[13] = halt_logit > state_logit ? 1 : 0;")
        emit("    }")

    emit("};")
    emit("")
    emit("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    emit('    py::class_<ENGINE_T>(m, "Engine", py::module_local())')
    emit("        .def(py::init<>())")
    emit("        .def(\"reset\", &ENGINE_T::reset)")
    emit("        .def(\"forward_f32\", &ENGINE_T::forward_f32)")
    if has_step:
        emit("        .def(\"machine_step\", &ENGINE_T::machine_step)")
    emit("        .def(\"insert_write_f32\", &ENGINE_T::insert_write_f32);")
    emit("}")

    source = "\n".join(lines) + "\n"
    return source, n_memory_heads, has_step


class GeneratedWeightsEngine:
    """Drop-in for SparseWeightsEngine; exact int64 default, f32 replica."""

    def __init__(self, model, *, dtype: str = "float32",
                 step_spec: dict | None = None, arith: str = "int32"):
        if dtype != "float32":
            raise ValueError("the generated engine is float32-only; use "
                             "matmul='engine' for the float64 diagnostic path")
        self.dtype = dtype
        self.arith = arith
        source, self.n_memory_heads, self.has_machine_step = generate_source(
            model, step_spec, arith
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        source = source.replace("ENGINE_T", f"Engine_{digest}")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        src_path = os.path.join(_CACHE_DIR, f"engine_{digest}.cpp")
        if not os.path.exists(src_path):
            with open(src_path, "w") as f:
                f.write(source)

        from torch.utils.cpp_extension import load

        build_start = time.perf_counter()
        cflags = ["-O2", "-std=c++17"]
        if arith == "int32":
            # int32 exactness argument is mod-2^32: partial sums may wrap,
            # every final value fits.  Wrap must be defined behavior.
            cflags.append("-fwrapv")
        module = load(
            name=f"weights_gen_{digest}",
            sources=[src_path],
            extra_cflags=cflags,
            extra_include_paths=[_ATTENTION_DIR],
            verbose=False,
        )
        self.compile_seconds = time.perf_counter() - build_start
        self._engine = module.Engine()
        if self.has_machine_step:
            # Bound method exposed directly: the runner's hot loop calls it
            # once per instruction with no wrapper frame in between.
            self.machine_step = self._engine.machine_step

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
