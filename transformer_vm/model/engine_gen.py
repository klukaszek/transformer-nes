"""Generated straight-line C++ engine for the analytic transformer weights.

A generic CSR forward spends almost all its time on the dependent scalar gather -- loading indices[k]/data[k] and chasing them (~1.5-2 ns/nnz; the hulls are only ~8%).  This module applies the kernel.py trick to the weights: it emits one C++ translation unit specific to the built model, with column indices as immediates and +-1 coefficients as add/sub, so those loads disappear.

Three arithmetic modes:

- ``arith="int32"`` (default): the residual stream is exact fixed-point
  int32 at scale 2**F (F=1 covers the half-integer values the +-0.5 direct-map coefficients produce; every rescaling shift is guarded and throws rather than round).  Every residual-feeding coefficient is an integer, all lookup heads are LATEST tie-break so hull outputs return the values that were inserted, and the STATE/HALT head rows are integers -- the whole non-attention computation is exact integer arithmetic.  Partial sums may wrap mod 2^32 (-fwrapv); every final value fits, so results are exact.  Q/K/V and the position features live in double, feeding the hulls directly (hulls always resolve in double anyway), and clear_key-gated inserts (ky -= 1e30, provably never a query winner) are skipped entirely.

- ``arith="int64"``: the same emission at int64 width (no wrap argument needed); measured slower than int32 -- the linear part is load-bound, so value density wins.

- ``arith="f32"``: float32 arithmetic (CSR column order, left-to-right accumulation, float32 hexfloat literals); kept as a benchmarking reference for the int paths.

All modes generate the direct-WRITE key construction from graph.core's LATEST_ALPHA/KEY_OFFSET instead of hand-copying it, so the generated engine cannot silently desync from the graph.
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


def _read_mode_table() -> list[int]:
    """Shadow-decode table for MMIO read watches, byte -> mode code.

    0 when the instruction never reads a memory operand, else a code the
    generated shadow_read_target() uses to compute the operand address from
    (pc, x, y) and the RAM mirror.  Same construction as kernel.py's table
    (shared source of truth: spec.OPCODES / READS_MEMORY); the C++ run loop
    needs it to pause before a device read exactly as the kernel does.
    """
    from transformer_vm.cpu6502.spec import (
        MODE_ABSOLUTE,
        MODE_ABSOLUTE_X,
        MODE_ABSOLUTE_Y,
        MODE_INDIRECT_X,
        MODE_INDIRECT_Y,
        MODE_ZERO_PAGE,
        MODE_ZERO_PAGE_X,
        MODE_ZERO_PAGE_Y,
        OPCODES,
        READS_MEMORY,
    )

    code = {
        MODE_ZERO_PAGE: 1, MODE_ZERO_PAGE_X: 2, MODE_ZERO_PAGE_Y: 3,
        MODE_ABSOLUTE: 4, MODE_ABSOLUTE_X: 5, MODE_ABSOLUTE_Y: 6,
        MODE_INDIRECT_X: 7, MODE_INDIRECT_Y: 8,
    }
    table = [0] * 256
    for byte, op in OPCODES.items():
        if op.operation in READS_MEMORY and op.mode in code:
            table[byte] = code[op.mode]
    return table


def _cycles_table() -> list[int]:
    """Base cycle count per opcode byte (0 for unknown bytes); NMI costs 7.

    Base cycles only, no branch-taken or page-cross penalties -- identical to
    kernel.py / nes.bus, so a run-loop cycle budget paces frames the same way.
    """
    from transformer_vm.cpu6502.spec import OPCODES

    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = op.cycles
    return table


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

    Accumulates in CSR column order (`acc += data[k] * x[indices[k]]`):
    C++ `+`/`-` are left-associative, so a single chained expression
    rounds identically to that loop.
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


def _csr(array) -> sp.csr_matrix:
    m = sp.csr_matrix(array)
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


def _check_ffn_kinds(model, layer_idx):
    """Validate the fusion contract behind ffn_kinds == 2.

    Kind-2 (linear passthrough/erase) neurons are folded into a direct
    residual map using only their value row, dropping the gate.  That is
    exact only if the gate is the constant-one expression, so a mislabeled
    neuron must fail here instead of silently computing the wrong activation.
    """
    w = model.ff_in[layer_idx].weight
    for j, kind in enumerate(model.ffn_kinds[layer_idx]):
        if kind != 2:
            continue
        gate = w[j]
        if gate[model.one_slot] != 1.0 or int((gate != 0).sum()) != 1:
            raise RuntimeError(
                f"L{layer_idx} ffn neuron {j}: kind-2 gate is not the constant one"
            )


# ── Static C++ blocks ───────────────────────────────────────────────────
# These sections do not depend on the built model (or depend only on the
# integer element type, spliced in for "ELEM").  Keeping them as readable
# C++ here, rather than hundreds of per-line emit() calls, is exact: the
# generated source is `"\n".join(lines)`, so emitting one multi-line block is
# byte-for-byte identical to emitting each of its lines.

_INCLUDES = """\
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include "hull2d_cht.h"

namespace py = pybind11;
using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LArr = py::array_t<long long, py::array::c_style | py::array::forcecast>;"""

# Fixed-point rescale/readout guards; ELEM is the int residual type.
_FIXED_POINT_HELPERS = """\
    static ELEM shr(ELEM v) {
        // Rescale a product of two fixed-point values; the
        // represented result must stay on the F-bit grid.
        if (v & (SCALE - 1))
            throw std::runtime_error("value finer than the fixed-point scale");
        return v >> F;
    }

    static long long shr2(ELEM v) {
        // Readout: trace-visible values are whole integers.
        if (v & (SCALE - 1))
            throw std::runtime_error("fractional readout value");
        return (long long)(v >> F);
    }

    static ELEM half(ELEM v) {
        // 0.5 coefficients in the fused direct maps: exact only
        // on even operands, and the invariant says they always
        // are -- fail loudly instead of rounding.
        if (v & 1)
            throw std::runtime_error("odd value under a 0.5 coefficient");
        return v >> 1;
    }"""

# Exact nearest-written-address argmax over one 4 KiB memory group.
_MEM_QUERY = """\
    // Exact replacement for the memory-group hull argmax: the
    // grouped parabola score -(q-a)^2 + q^2 + alpha*ilp(seq) is
    // maximized by the nearest written address; recency breaks
    // equidistant ties toward the later write, and below the
    // double ulp the hull's LATEST tie-break agrees.  One shared
    // structure per 4 KiB group serves every layer's heads (all
    // per-layer memory hulls held identical data).
    int mem_val[16][4096];
    long long mem_seq[16][4096];
    unsigned long long mem_bits[16][64];

    void mem_query(const int g, const long long q, long long out[2]) const {
        const unsigned long long* bits = mem_bits[g];
        long long below = -1, above = -1;
        if (q >= 0) {
            const long long start = q < 4095 ? q : 4095;
            int w = (int)(start >> 6);
            unsigned long long m = bits[w] & (~0ULL >> (63 - (start & 63)));
            while (true) {
                if (m) { below = ((long long)w << 6) + (63 - __builtin_clzll(m)); break; }
                if (--w < 0) break;
                m = bits[w];
            }
        }
        if (q <= 4095) {
            const long long start = q > 0 ? q : 0;
            int w = (int)(start >> 6);
            unsigned long long m = bits[w] & (~0ULL << (start & 63));
            while (true) {
                if (m) { above = ((long long)w << 6) + __builtin_ctzll(m); break; }
                if (++w > 63) break;
                m = bits[w];
            }
        }
        long long win;
        if (below < 0 && above < 0) { out[0] = 0; out[1] = 0; return; }
        else if (below < 0) win = above;
        else if (above < 0) win = below;
        else {
            const long long db = q - below, da = above - q;
            if (db < da) win = below;
            else if (da < db) win = above;
            else win = mem_seq[g][below] > mem_seq[g][above] ? below : above;
        }
        out[0] = mem_val[g][win];
        out[1] = ((long long)g << 12) + win;
    }"""

# Shadow-decode + the native run loop (KernelBus shape) + machine API.
_RUN_LOOP = """\
    // Shadow-decode the instruction at st_pc; return the address it
    // will read, or -1 (mirrors nes/bus read_target and the kernel).
    int shadow_read_target() const {
        const int mode = READ_MODE[ram[st_pc]];
        if (mode == 0) return -1;
        const int lo = ram[(st_pc + 1) & 0xFFFF];
        const int hi = ram[(st_pc + 2) & 0xFFFF];
        switch (mode) {
            case 1: return lo;
            case 2: return (lo + st_x) & 0xFF;
            case 3: return (lo + st_y) & 0xFF;
            case 4: return lo | (hi << 8);
            case 5: return ((lo | (hi << 8)) + st_x) & 0xFFFF;
            case 6: return ((lo | (hi << 8)) + st_y) & 0xFFFF;
            case 7: { const int zp = (lo + st_x) & 0xFF;
                      return ram[zp] | (ram[(zp + 1) & 0xFF] << 8); }
            case 8: { const int base = ram[lo] | (ram[(lo + 1) & 0xFF] << 8);
                      return (base + st_y) & 0xFFFF; }
        }
        return -1;
    }

    static bool in_ranges(const std::vector<std::pair<int,int>>& v, int addr) {
        for (const auto& r : v) if (addr >= r.first && addr <= r.second) return true;
        return false;
    }

    // Run up to max_steps instructions natively, pausing back to
    // Python only on halt / jump-to-self trap / watched device
    // access / cycle budget -- the same protocol and result dict as
    // the C++ kernel's run(), so KernelBus can drive this engine.
    py::dict run(long max_steps, bool stop_on_trap, long max_cycles) {
        std::vector<int> t_pc, t_a, t_x, t_y, t_s, t_p, t_ir, t_oplo, t_ophi, t_ea, t_data, t_halted;
        std::vector<long> w_row; std::vector<int> w_addr, w_value;
        std::vector<int> hit_addr, hit_value;
        const char* reason = "budget";
        int read_addr = -1;
        long cycles_this = 0;
        long long o[21];
        for (long i = 0; i < max_steps; ++i) {
            if (st_halted) { reason = "halt"; break; }
            const bool skip = resume_read_skip;
            resume_read_skip = false;
            if (!nmi_pending && !skip && !rwatch.empty()) {
                const int target = shadow_read_target();
                if (target >= 0 && in_ranges(rwatch, target)) {
                    resume_read_skip = true; read_addr = target; reason = "watch_read"; break;
                }
            }
            const bool was_nmi = nmi_pending;
            const int prev_pc = st_pc;
            step_core(o);  // updates st_*, applies writes to ram + memory
            t_pc.push_back((int)o[0]); t_a.push_back((int)o[1]);
            t_x.push_back((int)o[2]); t_y.push_back((int)o[3]);
            t_s.push_back((int)o[4]); t_p.push_back((int)o[5]);
            t_ir.push_back((int)o[6]); t_oplo.push_back((int)o[7]);
            t_ophi.push_back((int)o[8]); t_ea.push_back((int)o[9]);
            t_data.push_back((int)o[10]);
            const bool halted = o[11] != 0;
            t_halted.push_back(halted ? 1 : 0);
            const int n_writes = (int)o[14];
            int n_hits = 0, sh_addr[3], sh_value[3];
            for (int w = 0; w < n_writes; ++w) {
                const int wa = (int)o[15 + 2 * w], wv = (int)o[16 + 2 * w];
                w_row.push_back(step_index); w_addr.push_back(wa); w_value.push_back(wv);
                if (!wwatch.empty() && in_ranges(wwatch, wa)) {
                    sh_addr[n_hits] = wa; sh_value[n_hits] = wv; ++n_hits;
                }
            }
            ++step_index;
            const long step_cycles = was_nmi ? 7 : (long)CYCLES[(unsigned char)o[6]];
            cycles_this += step_cycles; cycles_total += step_cycles;
            for (int k = 0; k < n_hits; ++k) { hit_addr.push_back(sh_addr[k]); hit_value.push_back(sh_value[k]); }
            if (halted) { st_halted = true; reason = "halt"; break; }
            if (o[13]) { st_halted = true; reason = "halt"; break; }
            if (stop_on_trap && (int)o[12] == prev_pc) { reason = "trap"; break; }
            if (n_hits > 0) { reason = "watch_write"; break; }
            if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }
        }
        py::dict out;
        auto arr = [](const std::vector<int>& src) {
            return py::array_t<int>((py::ssize_t)src.size(), src.data());
        };
        out["pc"] = arr(t_pc);
        out["a"] = arr(t_a);
        out["x"] = arr(t_x);
        out["y"] = arr(t_y);
        out["s"] = arr(t_s);
        out["p"] = arr(t_p);
        out["ir"] = arr(t_ir);
        out["oplo"] = arr(t_oplo);
        out["ophi"] = arr(t_ophi);
        out["ea"] = arr(t_ea);
        out["data"] = arr(t_data);
        out["halted"] = arr(t_halted);
        out["w_row"] = py::array_t<long>((py::ssize_t)w_row.size(), w_row.data());
        out["w_addr"] = arr(w_addr);
        out["w_value"] = arr(w_value);
        out["hit_addr"] = arr(hit_addr);
        out["hit_value"] = arr(hit_value);
        out["read_addr"] = read_addr;
        out["reason"] = reason;
        out["cycles"] = cycles_this;
        return out;
    }

    // ── Machine API (mirrors the C++ kernel) ─────────────────
    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void watch_write(int lo, int hi) { wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void clear_watches() { rwatch.clear(); wwatch.clear(); }
    void set_nmi() { nmi_pending = true; }
    void inject_write(int addr, int value) { insert_write_f32(addr & 0xFFFF, value & 0xFF, seq + 1); }
    int read_ram(int addr) { return ram[addr & 0xFFFF]; }
    py::bytes dump_ram() { return py::bytes((const char*)ram, 65536); }

    void set_run_state(int pc, int a, int x, int y, int s, int p,
                       int ir, int oplo, int ophi, int ea, int data, int halted) {
        st_pc = pc & 0xFFFF; st_a = a & 0xFF; st_x = x & 0xFF; st_y = y & 0xFF;
        st_s = s & 0xFF; st_p = p & 0xFF;
        st_ir = ir & 0xFF; st_oplo = oplo & 0xFF; st_ophi = ophi & 0xFF;
        st_ea = ea & 0xFFFF; st_data = data & 0xFF;
        st_halted = halted != 0;
    }

    py::dict get_run_state() {
        py::dict st;
        st["pc"] = st_pc; st["a"] = st_a; st["x"] = st_x; st["y"] = st_y;
        st["s"] = st_s; st["p"] = st_p; st["halted"] = st_halted;
        st["ir"] = st_ir; st["oplo"] = st_oplo; st["ophi"] = st_ophi;
        st["ea"] = st_ea; st["data"] = st_data;
        st["steps"] = step_index; st["cycles"] = cycles_total;
        st["pos"] = (long)(seq + 1);
        return st;
    }"""

# The pybind .def() lines gated on has_step (the machine_step + run-loop API).
_PYBIND_STEP_METHODS = """\
        .def("machine_step", &ENGINE_T::machine_step)
        .def("run", &ENGINE_T::run, py::arg("max_steps"),
             py::arg("stop_on_trap") = false, py::arg("max_cycles") = 0L)
        .def("watch_read", &ENGINE_T::watch_read)
        .def("watch_write", &ENGINE_T::watch_write)
        .def("clear_watches", &ENGINE_T::clear_watches)
        .def("set_nmi", &ENGINE_T::set_nmi)
        .def("inject_write", &ENGINE_T::inject_write)
        .def("read_ram", &ENGINE_T::read_ram)
        .def("dump_ram", &ENGINE_T::dump_ram)
        .def("set_run_state", &ENGINE_T::set_run_state)
        .def("get_run_state", &ENGINE_T::get_run_state)"""


def generate_source(model, step_spec: dict | None = None,
                    arith: str = "int64", experts=None) -> tuple[str, int, bool]:
    """Emit the C++ source; returns (source, n_memory_heads, has_step).

    step_spec, if given, additionally emits a machine_step member that runs
    one whole 6502 instruction in C++ -- STATE-token embedding, position
    features, the layer stack, readout, write extraction with direct hull
    inserts, and the STATE/HALT head logits -- writing results into a
    caller-provided int64 buffer.  Layout: {"inputs": {name: slot},
    "p_flags": [(slot, mask)], "outputs": {name: slot},
    "halt_head"/"state_head": (cols, vals)}.

    experts, if given, is a list of opcode-routed experts.  Each is a dict
    ``{"name": "4C", "byte": 0x4C, "keep_neurons": {li: set[int]},
    "keep_heads": {li: set[int]}}`` giving, per layer, which ReGLU neurons and
    which lookup heads survive when the fetched opcode is pinned to ``byte``
    (an opcode-routed MoE: the decode one-hot the graph already computes is
    hoisted into a hard router).  A specialized ``step_core__<name>`` runs the
    pruned layer stack -- a residual row keeps only its live columns, and a row
    with no live column is dropped -- while the run loop / machine_step dispatch
    on ``ram[st_pc]`` (== the fetched opcode) and fall back to the full union
    step for opcodes without an expert and for NMIs.  The pruning is safe by
    construction: passthrough/erase slot bookkeeping is kept whole, memory is
    updated unconditionally through insert_write, and the running-argmax
    registers are self-correcting (a step that reads a register also folds the
    current, highest-position token, so a step that pruned an earlier fold
    cannot leave a stale winner).
    """
    experts = experts or []
    if KEY_OFFSET != 0:
        raise RuntimeError(
            "generated engine assumes KEY_OFFSET == 0 in the lookup key "
            "construction; regenerate the emission for a shifted key"
        )
    if arith not in ("int64", "int32", "f32"):
        raise ValueError(f"unknown arith {arith!r}")
    is_int = arith in ("int64", "int32")

    d_model = model.tok.weight.shape[1]
    n_heads = model.attn[0].num_heads
    n_layers = len(model.attn)

    lines: list[str] = []
    emit = lines.append

    body: list[str] = []  # per-layer member functions
    hull_tiebreak: list[int] = []  # flat hull id -> latest?
    mem_hulls: list[list[int]] = [[] for _ in range(16)]  # group -> flat hull ids
    mem_fallback_hulls: list[int] = []  # memory hulls kept on the hull path
    n_memory_heads = 0
    n_regs = 0  # running-argmax registers for constant-query heads
    n_hull_heads = 0  # heads still on real hulls

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

        def row_raw(indices, data, ref, where):
            return _expr_int(indices, data, ref, where)
    else:
        elem, zero = "float", "0.0f"

        def xref(i):
            return f"x[{i}]"

        def dref(i):  # unused in f32 mode
            return f"x[{i}]"

        def row_int(csr_m, r, ref, where):
            return _expr(*_row(csr_m, r), ref)

        def row_raw(indices, data, ref, where):
            return _expr(indices, data, ref)

    layer_sig = f"{elem}* x, const double* pf" if is_int else "float* x"

    layer_plans: list[dict] = []
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

        # Structural passthrough/erase fusion: fold kind-2 heads and neurons
        # into direct D->D residual maps (they never need a hull/ReGLU).
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

        # Per-head emission blocks (each self-contained: it defines ho{2h}
        # and ho{2h+1}).  Collected keyed by head so a routed expert can skip
        # a head wholesale when it produces no live value; the id assignment
        # (hull ids, running-argmax register indices) happens here once and is
        # shared by the union body and every expert body.
        head_blocks: dict[int, list[str]] = {}

        # ── Attention: hull heads, then residual updates ─────────────
        for h in lookup:
            b = head_blocks.setdefault(h, [])
            hid = hull_of_head[h]
            tb = "TieBreak::LATEST" if tiebreak[h] else "TieBreak::AVERAGE"
            # Memory heads qualify for the structure when the query is a
            # positive scalar multiple of [address - lo, 1]: qy is a lone
            # positive constant on the one-slot (the hardmax temperature),
            # and qx / qy has integer coefficients.  A positive scale
            # multiplies every score, so the argmax is unchanged.
            q1_idx, q1_dat = _row(ip, 2 * h + 1)
            mem_ok = (
                is_int
                and mem_groups[h] >= 0
                and len(q1_idx) == 1
                and int(q1_idx[0]) == model.one_slot
                and float(q1_dat[0]) > 0.0
            )
            if mem_ok:
                q_scale = float(q1_dat[0])
                q_idx, q_dat = _row(ip, 2 * h)
                q_int = np.asarray(q_dat, dtype=np.float64) / q_scale
                q_round = np.round(q_int)
                if not (np.all(np.abs(q_int - q_round) < 1e-6)
                        and np.all(np.abs(q_round) < 2**24)
                        and not any(int(i) <= 2 for i in q_idx)):
                    mem_ok = False

            # Non-memory heads with a CONSTANT query [QX, QY] (both rows a
            # lone term on the one-slot): each token's score QX*kx + QY*ky
            # is a fixed function of its key, so the historical argmax can
            # be maintained incrementally at insert time.  Replace-on->=
            # reproduces the LATEST tie-break (ilp is monotone in double,
            # so a later equal-key token never compares below an earlier
            # one), and cleared tokens are skipped as before.
            reg_ok = False
            v0_i = v1_i = None
            if is_int and not mem_ok and tiebreak[h]:
                q0_idx, q0_dat = _row(ip, 2 * h)
                reg_ok = (
                    len(q0_idx) == 1 and int(q0_idx[0]) == model.one_slot
                    and len(q1_idx) == 1 and int(q1_idx[0]) == model.one_slot
                )
                if reg_ok:
                    try:
                        v0_i = _expr_int(*_row(ip, 2 * d_model + 2 * h),
                                         xref, f"L{li} reg v0")
                        v1_i = _expr_int(*_row(ip, 2 * d_model + 2 * h + 1),
                                         xref, f"L{li} reg v1")
                    except RuntimeError:
                        reg_ok = False

            if mem_ok:
                # Memory-group lookup served by the exact nearest-address
                # structure: with qy == 1, the parabola score makes the
                # argmax the nearest written address, recency breaking
                # equidistant ties toward the later write (below the
                # double ulp the hull's LATEST tie-break picks the same
                # winner).  Writes arrive only via insert_write; the hull
                # stays empty and its guard makes that contract loud.
                g = mem_groups[h]
                qi_expr = _expr_int(q_idx, q_round, xref, f"L{li} mem query")
                k1 = _expr_dbl(*_row(ip, d_model + 2 * h + 1), dref)
                b.append(f"        {elem} ho{2 * h}, ho{2 * h + 1};")
                b.append("        {")
                b.append(f"            if (hull_seen[{hid}] != seq) {{")
                b.append(f"                const double kyg = {k1};")
                b.append("                if (kyg > -1.0e29)")
                b.append('                    throw std::runtime_error('
                         '"live forward insert into a memory-group hull");')
                b.append(f"                hull_seen[{hid}] = seq;")
                b.append("            }")
                b.append(f"            const long long qi = shr2({qi_expr});")
                b.append("            long long mo[2];")
                b.append(f"            mem_query({g}, qi, mo);")
                b.append(f"            ho{2 * h} = ({elem})(mo[0] << F);")
                b.append(f"            ho{2 * h + 1} = ({elem})(mo[1] << F);")
                b.append("        }")
            elif reg_ok:
                r = n_regs
                n_regs += 1
                qx = float(q0_dat[0])
                qy = float(q1_dat[0])
                k0 = _expr_dbl(*_row(ip, d_model + 2 * h), dref)
                k1 = _expr_dbl(*_row(ip, d_model + 2 * h + 1), dref)
                b.append(f"        {elem} ho{2 * h}, ho{2 * h + 1};")
                b.append("        {")
                b.append(f"            const double kyv = {k1};")
                b.append("            if (kyv > -1.0e29) {  // live: fold into the running argmax")
                b.append(f"                const double s = {_lit_d(qx)}*({k0}) + {_lit_d(qy)}*kyv;")
                b.append(f"                if (!reg_any[{r}] || s >= reg_s[{r}]) {{")
                b.append(f"                    reg_any[{r}] = true;")
                b.append(f"                    reg_s[{r}] = s;")
                b.append(f"                    reg_v0[{r}] = {v0_i};")
                b.append(f"                    reg_v1[{r}] = {v1_i};")
                b.append("                }")
                b.append("            }")
                b.append(f"            ho{2 * h} = reg_any[{r}] ? reg_v0[{r}] : 0;")
                b.append(f"            ho{2 * h + 1} = reg_any[{r}] ? reg_v1[{r}] : 0;")
                b.append("        }")
            elif is_int:
                n_hull_heads += 1
                if mem_groups[h] >= 0:
                    mem_fallback_hulls.append(hid)
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

        # attn_direct reads pre-attention x; capture before updating.  These
        # fused rows are passthrough/erase (slot-hygiene) work; they are kept
        # whole in every body -- a routed expert prunes value-producing units
        # (hull heads, ReGLU neurons), never the slot bookkeeping.
        ad_rows = [r for r in range(d_model)
                   if attn_direct.indptr[r + 1] > attn_direct.indptr[r]]
        ad_def = [f"        const {elem} ad{r} = "
                  f"{row_int(attn_direct, r, xref, f'L{li} attn_direct')};"
                  for r in ad_rows]
        # out_proj rows read the hull outputs ho{..}.  A routed expert keeps
        # only the columns whose head survives; a head column is dropped only
        # when the head is dead (produces no live value), so the term it would
        # contribute is unused.
        opkept: list[tuple[int, list[int], list[float]]] = []
        for r in range(d_model):
            if op_kept.indptr[r + 1] > op_kept.indptr[r]:
                idxs, dat = _row(op_kept, r)
                opkept.append((r, [int(c) for c in idxs], list(dat)))
        ad_apply = [f"        x[{r}] += ad{r};" for r in ad_rows]

        # ── FFN: direct map and activations read post-attention x ────
        fd_rows = [r for r in range(d_model)
                   if ffn_direct.indptr[r + 1] > ffn_direct.indptr[r]]
        fd_def = [f"        const {elem} fd{r} = "
                  f"{row_int(ffn_direct, r, xref, f'L{li} ffn_direct')};"
                  for r in fd_rows]
        neuron_blocks: dict[int, list[str]] = {}
        for j in live:
            g = row_int(fi, j, xref, f"L{li} ff_gate")
            v = row_int(fi, ffn + j, xref, f"L{li} ff_value")
            nb = [f"        {elem} a{j};"]
            if is_int:
                nb.append(f"        {{ const {elem} g = {g}; "
                          f"const {elem} v = {v}; "
                          f"a{j} = g > 0 ? shr(g*v) : 0; }}")
            else:
                nb.append(f"        {{ const {elem} g = {g}; const {elem} v = {v}; "
                          f"a{j} = g > {zero} ? g*v : {zero}; }}")
            neuron_blocks[j] = nb
        # ff_out rows read the ReGLU activations a{..}.  A routed expert drops
        # the columns whose neuron folds to zero for that opcode (an exact
        # elimination -- the dropped terms are provably 0); the row's remaining
        # terms and its target slot are untouched.
        fokept: list[tuple[int, list[int], list[float]]] = []
        for r in range(d_model):
            if fo_kept.indptr[r + 1] > fo_kept.indptr[r]:
                idxs, dat = _row(fo_kept, r)
                fokept.append((r, [int(c) for c in idxs], list(dat)))
        fd_apply = [f"        x[{r}] += fd{r};" for r in fd_rows]

        layer_plans.append({
            "li": li, "lookup": lookup, "head_blocks": head_blocks,
            "ad_def": ad_def, "opkept": opkept, "ad_apply": ad_apply,
            "fd_def": fd_def, "live": live, "neuron_blocks": neuron_blocks,
            "fokept": fokept, "fd_apply": fd_apply,
        })

    # ── Emit the layer functions: the union body plus one pruned body per
    # routed expert.  keep_heads / keep_neurons are None for the union
    # (everything kept) or the expert's per-layer survivor sets otherwise; a
    # residual row keeps only its live columns and is dropped when none
    # survive (exact term elimination). ──
    def emit_layer_fn(plan, suffix, keep_heads, keep_neurons):
        li = plan["li"]
        out = [f"    void layer_{li}{suffix}({layer_sig}) {{"]
        for h in plan["lookup"]:
            if keep_heads is None or h in keep_heads:
                out.extend(plan["head_blocks"][h])
        out.extend(plan["ad_def"])
        for r, idxs, dat in plan["opkept"]:
            if keep_heads is None:
                ki, kc = idxs, dat
            else:
                pairs = [(i, c) for i, c in zip(idxs, dat, strict=True)
                         if i // 2 in keep_heads]
                ki = [i for i, _ in pairs]
                kc = [c for _, c in pairs]
                if not ki:
                    continue
            out.append(f"        x[{r}] += "
                       f"{row_raw(ki, kc, lambda i: f'ho{i}', f'L{li} out_proj')};")
        out.extend(plan["ad_apply"])
        out.extend(plan["fd_def"])
        for j in plan["live"]:
            if keep_neurons is None or j in keep_neurons:
                out.extend(plan["neuron_blocks"][j])
        for r, idxs, dat in plan["fokept"]:
            if keep_neurons is None:
                ki, kc = idxs, dat
            else:
                pairs = [(i, c) for i, c in zip(idxs, dat, strict=True)
                         if i in keep_neurons]
                ki = [i for i, _ in pairs]
                kc = [c for _, c in pairs]
                if not ki:
                    continue
            out.append(f"        x[{r}] += "
                       f"{row_raw(ki, kc, lambda i: f'a{i}', f'L{li} ff_out')};")
        out.extend(plan["fd_apply"])
        out.append("    }")
        return out

    for plan in layer_plans:
        body.extend(emit_layer_fn(plan, "", None, None))
    for exp in experts:
        for plan in layer_plans:
            body.extend(emit_layer_fn(
                plan, f"__{exp['name']}",
                exp["keep_heads"].get(plan["li"], set()),
                exp["keep_neurons"].get(plan["li"], set()),
            ))

    n_hulls = len(hull_tiebreak)
    # The C++ run loop (KernelBus shape) needs the same memory-write path the
    # machine_step uses, so it is emitted under the same condition.
    has_run = step_spec is not None and n_memory_heads > 0

    emit(f"// Generated straight-line {arith} engine (model/engine_gen.py).")
    if is_int:
        emit("// Residual stream is exact int64 (integer addition is")
        emit("// associative, so the compiler may tree-reduce/vectorize);")
        emit("// Q/K/V and position features are double, hull-resolved.")
    else:
        emit("// float32 arithmetic: CSR column order, left-to-right")
        emit("// accumulation, float32 hexfloat literals.")
    emit(_INCLUDES)
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
        emit(_FIXED_POINT_HELPERS.replace("ELEM", elem))
        emit("")
        emit(_MEM_QUERY)
        if n_regs:
            emit("")
            emit("    // Running-argmax registers for constant-query lookup")
            emit("    // heads: the score of each token is a fixed function")
            emit("    // of its key, so the argmax is maintained at insert")
            emit("    // time and a query is a register read.")
            emit(f"    double reg_s[{n_regs}];")
            emit(f"    {elem} reg_v0[{n_regs}], reg_v1[{n_regs}];")
            emit(f"    bool reg_any[{n_regs}];")
    if has_run:
        # Persistent-machine state for the C++ run loop, mirroring the C++
        # kernel: a 64 KiB RAM mirror (for MMIO shadow decode), the
        # architectural registers, pending-NMI/read-skip flags, counters,
        # watch ranges, and the base-cycle / read-mode tables.
        read_mode = ", ".join(str(v) for v in _read_mode_table())
        cycles_tab = ", ".join(str(v) for v in _cycles_table())
        emit("")
        emit("    unsigned char ram[65536];  // mirror of the write log, for shadow decode")
        emit("    int st_pc = 0, st_a = 0, st_x = 0, st_y = 0, st_s = 0xFD, st_p = 0x24;")
        emit("    int st_ir = 0, st_oplo = 0, st_ophi = 0, st_ea = 0, st_data = 0;")
        emit("    bool st_halted = false;")
        emit("    bool nmi_pending = false;")
        emit("    bool resume_read_skip = false;")
        emit("    long step_index = 0;")
        emit("    long cycles_total = 0;")
        emit("    std::vector<std::pair<int,int>> rwatch, wwatch;")
        emit(f"    static constexpr unsigned char READ_MODE[256] = {{ {read_mode} }};")
        emit("    // Base cycles per opcode (no penalty cycles); NMI costs 7.")
        emit(f"    static constexpr unsigned char CYCLES[256] = {{ {cycles_tab} }};")
    emit("")
    emit("    ENGINE_T() { reset(); }")
    emit("")
    emit("    void reset() {")
    emit(f"        for (int i = 0; i < {n_hulls}; ++i) {{")
    emit("            hulls[i].clear();")
    emit("            hull_seen[i] = -1;")
    emit("        }")
    if is_int:
        emit("        for (int g = 0; g < 16; ++g)")
        emit("            for (int w = 0; w < 64; ++w) mem_bits[g][w] = 0ULL;")
    if n_regs:
        emit(f"        for (int r = 0; r < {n_regs}; ++r) reg_any[r] = false;")
    if has_run:
        emit("        for (int i = 0; i < 65536; ++i) ram[i] = 0;")
        emit("        st_pc = 0; st_a = st_x = st_y = 0; st_s = 0xFD; st_p = 0x24;")
        emit("        st_ir = st_oplo = st_ophi = st_ea = st_data = 0;")
        emit("        st_halted = false; nmi_pending = false; resume_read_skip = false;")
        emit("        step_index = 0; cycles_total = 0;")
        emit("        rwatch.clear(); wwatch.clear();")
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
    offs = [0]
    flat: list[int] = []
    fb_set = set(mem_fallback_hulls)
    fb_offs = [0]
    fb_flat: list[int] = []
    for g in range(16):
        flat.extend(mem_hulls[g])
        offs.append(len(flat))
        fb_flat.extend(i for i in mem_hulls[g] if i in fb_set)
        fb_offs.append(len(fb_flat))

    emit("    void insert_write_f32(int addr, int value, long pos) {")
    emit("        ++seq;")
    emit("        if (seq != pos)")
    emit('            throw std::runtime_error("direct WRITE position does not match engine sequence");')
    if has_run:
        emit("        ram[addr & 0xFFFF] = (unsigned char)(value & 0xFF);  // shadow-decode mirror")
    emit("        const int group = (addr >> 12) & 0xF;")
    if is_int:
        # Structure-served memory: a write is three stores; no key math.
        emit("        const int wl = addr & 0xFFF;")
        emit("        mem_val[group][wl] = value & 0xFF;")
        emit("        mem_seq[group][wl] = pos;")
        emit("        mem_bits[group][wl >> 6] |= 1ULL << (wl & 63);")
    hull_targets = fb_flat if is_int else flat
    hull_offs = fb_offs if is_int else offs
    if hull_targets:
        emit("        const int local_i = addr - group * 4096;")
        emit("        const float local = (float)local_i;")
        emit("        const float kx_f = 2.0f * local;")
        emit("        const float inv_log_pos = (float)(1.0 / std::log(2.0) - 1.0 / std::log((double)pos + 2.0));")
        emit(f"        const float ky_f = -local * local + {_lit(LATEST_ALPHA)} * inv_log_pos;")
        emit("        const double key[2] = {(double)kx_f, (double)ky_f};")
        emit("        const double val[2] = {(double)(value & 0xFF), (double)(addr & 0xFFFF)};")
        emit(f"        static const int offs[17] = {{{', '.join(map(str, hull_offs))}}};")
        emit(f"        static const int ids[{len(hull_targets)}] = {{{', '.join(map(str, hull_targets))}}};")
        emit("        for (int i = offs[group]; i < offs[group + 1]; ++i) {")
        emit("            const int hid = ids[i];")
        emit("            if (hull_seen[hid] == seq) continue;")
        emit("            hulls[hid].insert(key, val, seq);")
        emit("            hull_seen[hid] = seq;")
        emit("        }")
    elif not is_int:
        emit("        (void)group;")
    emit("    }")

    has_step = step_spec is not None and bool(flat)
    if has_step:
        ins = step_spec["inputs"]
        outs = step_spec["outputs"]

        def out_slot(name):
            return outs[name]

        # Embedded values live on the fixed-point grid in int mode.
        if is_int:
            one_e, zero_l = "SCALE", "0"

            def sc(v, e=elem):
                return f"(({e})({v}) << F)"
        else:
            one_e, zero_l = "1.0f", "0.0f"

            def sc(v):
                return f"(float)({v})"

        emit("")
        emit("    // One whole 6502 instruction from the persistent st_* state:")
        emit("    // STATE embed -> layers -> readout -> direct WRITE inserts ->")
        emit("    // halt logits.  Fills the 21-slot out buffer (trace row + next")
        emit("    // state + writes) and advances st_* to the next instruction.")
        if not is_int:
            emit(f"    float xbuf[{d_model}];")
            emit("")
            emit("    static long long rd(float v) {")
            emit("        // Python round() is half-to-even; nearbyint under the")
            emit("        // default FE_TONEAREST matches it (llround does not).")
            emit("        return (long long)std::nearbyint((double)v);")
            emit("    }")
        emit("")
        rdv = (lambda s: f"shr2(x[{s}])") if is_int else (lambda s: f"rd(x[{s}])")
        hc, hv = step_spec["halt_head"]
        shc, shv = step_spec["state_head"]

        def emit_step_core(fn_name, layer_suffix):
            # One whole instruction, sharing the embed/readout/write/halt
            # protocol; layer_suffix selects the union stack ("") or a routed
            # expert's pruned stack ("__<name>").
            emit(f"    void {fn_name}(long long* out) {{")
            emit(f"        {elem}* x = xbuf;")
            emit(f"        for (int i = 0; i < {d_model}; ++i) x[i] = {zero};")
            emit(f"        x[{model.one_slot}] = {one_e};")
            emit(f"        x[{ins['is_state']}] = {one_e};")
            emit(f"        x[{ins['pc_lo']}] = {sc('st_pc & 0xFF')};")
            emit(f"        x[{ins['pc_hi']}] = {sc('(st_pc >> 8) & 0xFF')};")
            emit(f"        x[{ins['a']}] = {sc('st_a')};")
            emit(f"        x[{ins['x']}] = {sc('st_x')};")
            emit(f"        x[{ins['y']}] = {sc('st_y')};")
            emit(f"        x[{ins['s']}] = {sc('st_s')};")
            for slot, mask in step_spec["p_flags"]:
                emit(f"        x[{slot}] = (st_p & {mask}) ? {one_e} : {zero_l};")
            emit(f"        x[{ins['ir']}] = {sc('st_ir')};")
            emit(f"        x[{ins['oplo']}] = {sc('st_oplo')};")
            emit(f"        x[{ins['ophi']}] = {sc('st_ophi')};")
            emit(f"        x[{ins['ea_lo']}] = {sc('st_ea & 0xFF')};")
            emit(f"        x[{ins['ea_hi']}] = {sc('(st_ea >> 8) & 0xFF')};")
            emit(f"        x[{ins['data']}] = {sc('st_data')};")
            emit(f"        x[{ins['halted']}] = {sc('st_halted ? 1 : 0')};")
            emit(f"        x[{ins['nmi']}] = {sc('nmi_pending ? 1 : 0')};")
            emit("        nmi_pending = false;")
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
                emit(f"        layer_{li}{layer_suffix}(x{', pf' if is_int else ''});")
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
            if is_int:
                emit(f"        const long long halt_logit = "
                     f"{_expr_int(hc, hv, lambda i: f'x[{i}]', 'halt head')};")
                emit(f"        const long long state_logit = "
                     f"{_expr_int(shc, shv, lambda i: f'x[{i}]', 'state head')};")
            else:
                emit(f"        const float halt_logit = {_expr(hc, hv, lambda i: f'x[{i}]')};")
                emit(f"        const float state_logit = {_expr(shc, shv, lambda i: f'x[{i}]')};")
            emit("        out[13] = halt_logit > state_logit ? 1 : 0;")
            emit("        // Advance the architectural state (inputs are fully")
            emit("        // consumed by the embed above before this update).")
            emit("        st_pc = (int)out[12];")
            emit("        st_a = (int)out[1]; st_x = (int)out[2]; st_y = (int)out[3];")
            emit("        st_s = (int)out[4]; st_p = (int)out[5];")
            emit("        st_ir = (int)out[6]; st_oplo = (int)out[7]; st_ophi = (int)out[8];")
            emit("        st_ea = (int)out[9]; st_data = (int)out[10];")
            emit("        st_halted = out[11] != 0;")
            emit("    }")

        emit_step_core("step_core_union", "")
        for exp in experts:
            emit("")
            emit_step_core(f"step_core__{exp['name']}", f"__{exp['name']}")

        # Opcode-routed dispatch (the MoE router): the fetched opcode is
        # ram[st_pc] (== the graph's memory read at pc), so an expert built by
        # pinning is_op_<byte> runs exactly when that byte is fetched.  A
        # pending NMI overrides the instruction, so it takes the union step.
        emit("")
        emit("    void step_core(long long* out) {")
        if experts:
            emit("        if (!nmi_pending) {")
            emit("            switch (ram[st_pc]) {")
            for exp in experts:
                emit(f"                case 0x{exp['byte']:02X}: "
                     f"step_core__{exp['name']}(out); return;")
            emit("            }")
            emit("        }")
        emit("        step_core_union(out);")
        emit("    }")
        emit("")
        emit("    // Per-instruction entry point: seed st_* from arguments, run one")
        emit("    // step.  Kept so the Python per-step path is byte-identical.")
        emit("    void machine_step(int pc, int a, int xr, int yr, int sr,")
        emit("                      int p, int ir, int oplo, int ophi,")
        emit("                      int ea, int data, int halted, int nmi,")
        emit("                      LArr out_arr) {")
        emit('        if ((int)out_arr.size() != 21) throw std::runtime_error("step out buffer must have 21 slots");')
        emit("        st_pc = pc & 0xFFFF; st_a = a & 0xFF; st_x = xr & 0xFF;")
        emit("        st_y = yr & 0xFF; st_s = sr & 0xFF; st_p = p & 0xFF;")
        emit("        st_ir = ir & 0xFF; st_oplo = oplo & 0xFF; st_ophi = ophi & 0xFF;")
        emit("        st_ea = ea & 0xFFFF; st_data = data & 0xFF;")
        emit("        st_halted = halted != 0; nmi_pending = nmi != 0;")
        emit("        step_core(out_arr.mutable_data());")
        emit("    }")

        # ── C++ run loop (KernelBus shape) and machine API ───────────
        emit("")
        emit(_RUN_LOOP)

    emit("};")
    emit("")
    emit("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    emit('    py::class_<ENGINE_T>(m, "Engine", py::module_local())')
    emit("        .def(py::init<>())")
    emit("        .def(\"reset\", &ENGINE_T::reset)")
    emit("        .def(\"forward_f32\", &ENGINE_T::forward_f32)")
    if has_step:
        emit(_PYBIND_STEP_METHODS)
    emit("        .def(\"insert_write_f32\", &ENGINE_T::insert_write_f32);")
    emit("}")

    source = "\n".join(lines) + "\n"
    return source, n_memory_heads, has_step


class GeneratedWeightsEngine:
    """The 6502 transformer as generated C++; exact int32 residual by default."""

    def __init__(self, model, *, step_spec: dict | None = None,
                 arith: str = "int32", experts=None):
        self.dtype = "float32"
        self.arith = arith
        self.n_experts = len(experts or [])
        source, self.n_memory_heads, self.has_machine_step = generate_source(
            model, step_spec, arith, experts=experts
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
        # The run loop is emitted under the same condition as machine_step.
        self.has_run_loop = self.has_machine_step
        if self.has_machine_step:
            # Bound methods exposed directly: the hot loops call them with no
            # wrapper frame in between.
            self.machine_step = self._engine.machine_step
            self.run = self._engine.run
            self.set_run_state = self._engine.set_run_state
            self.get_run_state = self._engine.get_run_state
            self.watch_read = self._engine.watch_read
            self.watch_write = self._engine.watch_write
            self.clear_watches = self._engine.clear_watches
            self.set_nmi = self._engine.set_nmi
            self.engine_inject_write = self._engine.inject_write
            self.read_ram = self._engine.read_ram
            self.dump_ram = self._engine.dump_ram

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
