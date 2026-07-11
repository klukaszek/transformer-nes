"""Generated straight-line C++ engine for the analytic transformer weights.

A generic CSR forward spends almost all its time on the dependent scalar gather -- loading indices[k]/data[k] and chasing them (~1.5-2 ns/nnz; the hulls are only ~8%).  This module applies the kernel.py trick to the weights: it emits one C++ translation unit specific to the built model, with column indices as immediates and +-1 coefficients as add/sub, so those loads disappear.

Three arithmetic modes:

- ``arith="int32"`` (default): the residual stream is exact fixed-point
  int32 at scale 2**F (F=1 covers the half-integer values the +-0.5 direct-map coefficients produce; every rescaling shift is guarded and throws rather than round).  Every residual-feeding coefficient is an integer, all lookup heads are LATEST tie-break so hull outputs return the values that were inserted, and the STATE/HALT head rows are integers -- the whole non-attention computation is exact integer arithmetic.  Partial sums may wrap mod 2^32 (-fwrapv); every final value fits, so results are exact.  Q/K/V and the position features live in double, feeding the hulls directly (hulls always resolve in double anyway), and clear_key-gated inserts (ky -= 1e30, provably never a query winner) are skipped entirely.

- ``arith="int64"``: the same emission at int64 width (no wrap argument needed); measured slower than int32 -- the linear part is load-bound, so value density wins.

- ``arith="f32"``: float32 arithmetic (CSR column order, left-to-right accumulation, float32 hexfloat literals); kept as a benchmarking reference for the int paths.

The direct-WRITE key construction mirrors graph.core's `_to_2d_key` (kx = 2*local, ky = -local^2, exact ties resolved by the LATEST tie-break).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import scipy.sparse as sp

_ATTENTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention"
)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_engine_cache")
_MANIFEST_PATH = os.path.join(_CACHE_DIR, "manifest.json")

# Everything whose content can change the emitted engine, hashed into the
# config key so any edit is a cache miss (over-inclusion is safe).
_CODE_DIRS = ("graph", "scheduler", "model", "attention")
_CODE_FILES = (
    "cpu6502/spec.py",
    "cpu6502/evaluator.py",
    "cpu6502/reference.py",
    "cpu6502/weights_runner.py",
)
_code_digest_cache: str | None = None


def _code_digest() -> str:
    global _code_digest_cache
    if _code_digest_cache is not None:
        return _code_digest_cache
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = []
    for d in _CODE_DIRS:
        for root, dirs, files in os.walk(os.path.join(pkg, d)):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "_engine_cache")]
            paths.extend(
                os.path.join(root, f) for f in files
                if f.endswith((".py", ".h", ".hpp", ".cpp"))
            )
    paths.extend(os.path.join(pkg, *f.split("/")) for f in _CODE_FILES)
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: os.path.relpath(p, pkg)):
        h.update(os.path.relpath(p, pkg).encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    _code_digest_cache = h.hexdigest()
    return _code_digest_cache


def engine_config_key(*, bcd: bool, arith: str, expert_bytes, extra: str = "") -> str:
    """Cache key for a built engine, computable without folding or emitting:
    the config plus a digest of every source file that shapes the emission."""
    tag = ",".join(f"{int(b):02X}" for b in expert_bytes)
    raw = f"{_code_digest()}|bcd={int(bcd)}|arith={arith}|experts={tag}|{extra}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _manifest_read() -> dict:
    try:
        with open(_MANIFEST_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _manifest_put(key: str, entry: dict) -> None:
    manifest = _manifest_read()
    manifest[key] = entry
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = _MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, _MANIFEST_PATH)


_FUNC_RE = re.compile(r"^    void ((?:layer_|step_core)\w*)\((.*)\) \{$")


def _shard_source(source: str, struct_name: str, n_shards: int) -> tuple[str, list[str]]:
    """Split the single-TU engine into a header + n_shards bodies so ninja
    compiles them in parallel.  The heavy member functions (layer_*,
    step_core*) become out-of-class definitions, greedily balanced by line
    count; everything else stays in-class (implicitly inline) in the header.
    The PYBIND module block lands in shard 0.  Purely a deterministic
    post-processing of the emitted text -- the digest covers the original."""
    lines = source.split("\n")
    header: list[str] = ["#pragma once"]
    chunks: list[list[str]] = []
    pybind: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FUNC_RE.match(line)
        if m is not None:
            depth = line.count("{") - line.count("}")
            chunk = [f"void {struct_name}::{m.group(1)}({m.group(2)}) {{"]
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                chunk.append(lines[i])
                i += 1
            header.append(f"    void {m.group(1)}({m.group(2)});")
            chunks.append(chunk)
            continue
        if line.startswith("PYBIND11_MODULE("):
            depth = line.count("{") - line.count("}")
            pybind.append(line)
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                pybind.append(lines[i])
                i += 1
            continue
        header.append(line)
        i += 1

    shards = [[] for _ in range(max(1, n_shards))]
    sizes = [0] * len(shards)
    for chunk in sorted(chunks, key=len, reverse=True):
        k = sizes.index(min(sizes))
        shards[k].extend(chunk + [""])
        sizes[k] += len(chunk)
    shards[0] = pybind + [""] + shards[0]
    return "\n".join(header) + "\n", ["\n".join(s) + "\n" for s in shards]


def _load_engine_module(digest: str, arith: str) -> tuple[object, float]:
    """Import the compiled engine, preferring the .so cached next to the
    source (a direct dlopen, no ninja/clang -- the distributable path);
    compile from the cached source shards only when the .so is absent."""
    name = f"weights_gen_{digest}"
    if name in sys.modules:
        return sys.modules[name], 0.0
    build_dir = os.path.join(_CACHE_DIR, f"build_{digest}")
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.exists(so_path):
        spec = importlib.util.spec_from_file_location(name, so_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module, 0.0

    from torch.utils.cpp_extension import load

    build_start = time.perf_counter()
    cflags = ["-O2", "-std=c++17"]
    if arith == "int32":
        # int32 exactness argument is mod-2^32: partial sums may wrap,
        # every final value fits.  Wrap must be defined behavior.
        cflags.append("-fwrapv")
    os.makedirs(build_dir, exist_ok=True)
    shard_glob = sorted(
        f for f in os.listdir(_CACHE_DIR)
        if f.startswith(f"engine_{digest}_s") and f.endswith(".cpp")
    )
    module = load(
        name=name,
        sources=[os.path.join(_CACHE_DIR, f) for f in shard_glob],
        extra_cflags=cflags,
        extra_include_paths=[_ATTENTION_DIR, _CACHE_DIR],
        build_directory=build_dir,
        verbose=False,
    )
    return module, time.perf_counter() - build_start


def seqmod_config_key(*, core_digest: str, bcd: bool, arith: str, windows) -> str:
    """Cache key for a per-ROM sequence module.  The core digest covers the
    header the module compiles against; the code digest covers fold/emission
    changes that alter the module without altering the core's source."""
    tag = ",".join(f"{h:04X}:{bytes(c).hex()}x{r}" for h, c, r in windows)
    raw = (f"{_code_digest()}|core={core_digest}|bcd={int(bcd)}"
           f"|arith={arith}|seq={tag}")
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_seqmod_module(digest: str, arith: str) -> tuple[object, float]:
    """Import a compiled sequence module (single small TU), preferring the
    cached .so exactly like _load_engine_module."""
    name = f"seqmod_gen_{digest}"
    if name in sys.modules:
        return sys.modules[name], 0.0
    build_dir = os.path.join(_CACHE_DIR, f"build_{digest}")
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.exists(so_path):
        spec = importlib.util.spec_from_file_location(name, so_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module, 0.0

    from torch.utils.cpp_extension import load

    build_start = time.perf_counter()
    cflags = ["-O2", "-std=c++17"]
    if arith == "int32":
        cflags.append("-fwrapv")
    os.makedirs(build_dir, exist_ok=True)
    module = load(
        name=name,
        sources=[os.path.join(_CACHE_DIR, f"seqmod_{digest}.cpp")],
        extra_cflags=cflags,
        extra_include_paths=[_ATTENTION_DIR, _CACHE_DIR],
        build_directory=build_dir,
        verbose=False,
    )
    return module, time.perf_counter() - build_start


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
#include <cstdint>
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
        // Written addresses answer for themselves: below == above == q,
        // and the tie-break picks the same entry either way.
        if (q >= 0 && q <= 4095 && (bits[q >> 6] >> (q & 63)) & 1ULL) {
            out[0] = mem_val[g][q];
            out[1] = ((long long)g << 12) + q;
            return;
        }
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

# Per-ROM sequence-module hook: a separately compiled window module attaches
# its dispatch through a raw function pointer, so a new ROM's window set never
# recompiles this TU.  The module fills seq_out with K trace rows per forward.
# A pinned module window outranks the canonical shape table (it folded the
# operands too, so its forward is cheaper); both fall back to per-step.
_SEQ_HOOK = """\
    typedef int (*SeqFn)(void*, void*, long, long, long, long, int);
    SeqFn seq_fn = nullptr;
    void* seq_ctx = nullptr;
    static constexpr int SEQ_MAX_K = 32;
    long long seq_out[21 * SEQ_MAX_K];

    int seq_dispatch(long i, long max_steps, long cycles_this,
                     long max_cycles, bool stop_on_trap) {
        if (seq_fn) {
            const int k = seq_fn(seq_ctx, this, i, max_steps, cycles_this,
                                 max_cycles, stop_on_trap ? 1 : 0);
            if (k) return k;
        }
        return shape_dispatch(i, max_steps, cycles_this, max_cycles,
                              stop_on_trap);
    }

    void attach_seq(std::uintptr_t fn, std::uintptr_t ctx) {
        seq_fn = (SeqFn)fn; seq_ctx = (void*)ctx;
    }
    void detach_seq() { seq_fn = nullptr; seq_ctx = nullptr; }"""

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
                    if (target != lease_addr) {
                        resume_read_skip = true; read_addr = target; reason = "watch_read"; break;
                    }
                    // Leased: serve the device read natively, exactly
                    // what the Python pause path would inject.
                    inject_write(target, lease_value);
                }
            }
            // A routed sequence consumes a whole window in one forward;
            // its guards (in seq_dispatch) keep this step-for-step
            // equivalent to the per-instruction path below.
            {
                const int k_seq = seq_dispatch(i, max_steps, cycles_this,
                                               max_cycles, stop_on_trap);
                if (k_seq > 0) {
                    for (int k = 0; k < k_seq; ++k) {
                        const long long* ok = seq_out + 21 * k;
                        t_pc.push_back((int)ok[0]); t_a.push_back((int)ok[1]);
                        t_x.push_back((int)ok[2]); t_y.push_back((int)ok[3]);
                        t_s.push_back((int)ok[4]); t_p.push_back((int)ok[5]);
                        t_ir.push_back((int)ok[6]); t_oplo.push_back((int)ok[7]);
                        t_ophi.push_back((int)ok[8]); t_ea.push_back((int)ok[9]);
                        t_data.push_back((int)ok[10]);
                        t_halted.push_back((int)ok[11]);
                        ++step_index;
                        const long step_cycles = (long)CYCLES[(unsigned char)ok[6]];
                        cycles_this += step_cycles; cycles_total += step_cycles;
                    }
                    if (seq_out[21 * (k_seq - 1) + 13]) {
                        st_halted = true; reason = "halt"; break;
                    }
                    i += k_seq - 1;
                    if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }
                    continue;
                }
            }
            const bool was_nmi = nmi_pending;
            const int prev_pc = st_pc;
            ++fallback_count[prev_pc];  // JIT profile: per-instruction steps only
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

    // ── JIT profile counter ──────────────────────────────────
    // The run loop counts every per-instruction (fallback) step by pc;
    // jit_hot reports pcs past the CUMULATIVE threshold that have no
    // window yet, rearming only those (a rejected pc re-reports after
    // re-heating; the Python side's rejected set filters it cheaply).
    unsigned int fallback_count[65536];

    py::array_t<int> jit_hot(long min_count) {
        std::vector<int> hot;
        for (int pc = 0; pc < 65536; ++pc)
            if ((long)fallback_count[pc] >= min_count && !shape_of_pc[pc]) {
                hot.push_back(pc);
                fallback_count[pc] = 0;
            }
        return py::array_t<int>((py::ssize_t)hot.size(), hot.data());
    }

    int window_shape(int pc) { return shape_of_pc[pc & 0xFFFF]; }

    // ── Machine API (mirrors the C++ kernel) ─────────────────
    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void watch_write(int lo, int hi) { wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void clear_watches() { rwatch.clear(); wwatch.clear(); }
    void set_read_lease(int addr, int value) { lease_addr = addr & 0xFFFF; lease_value = value & 0xFF; }
    void clear_read_lease() { lease_addr = -1; }
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
        .def("set_read_lease", &ENGINE_T::set_read_lease)
        .def("clear_read_lease", &ENGINE_T::clear_read_lease)
        .def("set_nmi", &ENGINE_T::set_nmi)
        .def("inject_write", &ENGINE_T::inject_write)
        .def("read_ram", &ENGINE_T::read_ram)
        .def("dump_ram", &ENGINE_T::dump_ram)
        .def("set_run_state", &ENGINE_T::set_run_state)
        .def("get_run_state", &ENGINE_T::get_run_state)
        .def("attach_seq", &ENGINE_T::attach_seq)
        .def("detach_seq", &ENGINE_T::detach_seq)
        .def("add_window", &ENGINE_T::add_window)
        .def("clear_windows", &ENGINE_T::clear_windows)
        .def("jit_hot", &ENGINE_T::jit_hot)
        .def("window_shape", &ENGINE_T::window_shape)"""


# Core-side seq_out capacity; keep in sync with SEQ_MAX_K in _SEQ_HOOK.
_SEQ_MAX_K = 32


def _assemble_seq_module(lines, body, experts, core_digest, elem,
                         n_seen, emit_step_core) -> tuple[str, int, bool]:
    """Assemble the per-ROM sequence-module TU from the emitted expert
    bodies: a SEQMOD_T struct compiled against the core's generated header,
    exporting its dispatch as a raw function pointer for attach_seq()."""
    emit = lines.append
    core_name = f"Engine_{core_digest}"
    mod_xbuf = max(e["model"].tok.weight.shape[1] for e in experts)
    max_k = max(e["n_steps"] for e in experts)
    if max_k > _SEQ_MAX_K:
        raise RuntimeError(
            f"sequence window of {max_k} steps exceeds the core's "
            f"seq_out capacity ({_SEQ_MAX_K})"
        )
    emit("// Generated per-ROM sequence module (model/engine_gen.py): folded")
    emit("// window experts + dispatch, compiled against the core engine's")
    emit("// generated header and attached at runtime via attach_seq().  All")
    emit("// shared machine state is reached through the typed reference E;")
    emit("// only scratch (xbuf, the hull_seen guards) lives in the module.")
    emit(f'#include "engine_{core_digest}.hpp"')
    emit("")
    emit("struct SEQMOD_T {")
    emit(f"    using Eng = {core_name};")
    emit("    static constexpr int F = Eng::F;")
    emit(f"    static constexpr {elem} SCALE = Eng::SCALE;")
    emit("    static constexpr double SCALE_D = Eng::SCALE_D;")
    emit(f"    static {elem} shr({elem} v) {{ return Eng::shr(v); }}")
    emit(f"    static long long shr2({elem} v) {{ return Eng::shr2(v); }}")
    emit(f"    static {elem} half({elem} v) {{ return Eng::half(v); }}")
    emit(f"    {elem} xbuf[{mod_xbuf}];")
    emit("    // Guard state only: module memory heads never hold hull data")
    emit("    // (ids are shared with the discarded union planning pass, so")
    emit("    // the array is sparse -- a few hundred longs, not worth rebasing).")
    emit(f"    long hull_seen[{max(n_seen, 1)}];")
    emit(f"    SEQMOD_T() {{ for (int i = 0; i < {max(n_seen, 1)}; ++i) "
         "hull_seen[i] = -1; }")
    emit("")
    lines.extend(body)
    for exp in experts:
        emit("")
        emit_step_core(f"step_core__{exp['name']}", f"__{exp['name']}",
                       exp["model"], exp["step_spec"])
    # Window dispatch: same guards as the per-instruction path needs for
    # step-for-step equivalence -- pc at the head, every window byte
    # matched, no pending NMI, enough step/cycle budget for the window.
    emit("")
    emit("    int dispatch(Eng& E, long i, long max_steps, long cycles_this,")
    emit("                 long max_cycles, bool stop_on_trap) {")
    emit("        if (E.nmi_pending || stop_on_trap) return 0;")
    for exp in experts:
        head, window = exp["head"], exp["window"]
        k, prefix = exp["n_steps"], exp["cycles_prefix"]
        match = " && ".join(
            f"E.ram[0x{(head + j) & 0xFFFF:04X}] == 0x{b:02X}"
            for j, b in enumerate(window)
        )
        emit(f"        if (E.st_pc == 0x{head:04X} && i + {k} <= max_steps")
        emit(f"            && (max_cycles <= 0 || cycles_this + {prefix} < max_cycles)")
        emit(f"            && {match}) {{")
        emit(f"            step_core__{exp['name']}(E, E.seq_out);")
        emit(f"            return {k};")
        emit("        }")
    emit("        return 0;")
    emit("    }")
    emit("")
    emit("    static int dispatch_tramp(void* ctx, void* eng, long i,")
    emit("                              long max_steps, long cycles_this,")
    emit("                              long max_cycles, int stop_on_trap) {")
    emit("        return ((SEQMOD_T*)ctx)->dispatch(*(Eng*)eng, i, max_steps,")
    emit("                                          cycles_this, max_cycles,")
    emit("                                          stop_on_trap != 0);")
    emit("    }")
    emit("};")
    emit("")
    emit("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    emit('    py::class_<SEQMOD_T>(m, "SeqModule", py::module_local())')
    emit("        .def(py::init<>())")
    emit('        .def("ctx_ptr", [](SEQMOD_T& s) { return (std::uintptr_t)&s; })')
    emit('        .def_static("fn_ptr", []() { '
         "return (std::uintptr_t)&SEQMOD_T::dispatch_tramp; });")
    emit(f'    m.attr("core_name") = "{core_name}";')
    emit(f'    m.attr("max_k") = {max_k};')
    emit("}")
    return "\n".join(lines) + "\n", 0, False


def generate_source(model, step_spec: dict | None = None,
                    arith: str = "int64", experts=None,
                    seq_module_core: str | None = None) -> tuple[str, int, bool]:
    """Emit the C++ source; returns (source, n_memory_heads, has_step).

    step_spec, if given, additionally emits a machine_step member that runs
    one whole 6502 instruction in C++ -- STATE-token embedding, position
    features, the layer stack, readout, write extraction with direct hull
    inserts, and the STATE/HALT head logits -- writing results into a
    caller-provided int64 buffer.  Layout: {"inputs": {name: slot},
    "p_flags": [(slot, mask)], "outputs": {name: slot},
    "halt_head"/"state_head": (cols, vals)}.

    experts, if given, is a list of opcode-routed *rebuilt expert models*
    (a genuine top-1 hard-routed MoE).  Each is a dict ``{"name": "4C",
    "byte": 0x4C, "model": WeightModel, "step_spec": dict,
    "union_lookup_of": {expert_lu_id: union_lu_id}}`` -- a full transformer
    built from the union graph partially evaluated under the pinned decode
    one-hot (graph/fold.py), scheduled and weight-compiled independently, so
    a gate=1 mux fold really disappears instead of surviving as a union
    neuron.  Expert residual layouts are private; the interface between
    experts is the persistent machine state: the ``st_*`` registers, the RAM
    mirror, the 16x4KiB memory structure, and the running-argmax registers,
    which are keyed by (union lookup id, value pair) and resolved through
    the fold's provenance.  An expert head whose classification disagrees
    with the union's (a memory head failing structure eligibility, a
    constant-query head with no union register, or a head needing a real
    hull) is a hard generation error: a per-expert hull would accumulate
    divergent history across routed steps.  The run loop / machine_step
    dispatch on ``ram[st_pc]`` (== the fetched opcode; the fold's pin) and
    fall back to the union step for cold opcodes and pending NMIs.

    seq_module_core, if given, is the digest of an already-emitted core
    engine: instead of a full engine, emit a tiny per-ROM *sequence module*
    -- the given kind="seq" experts compiled against the core's generated
    header, reaching all shared machine state (st_* registers, RAM mirror,
    memory structure, running-argmax registers) through a typed engine
    reference, and exporting its window dispatch as a raw function pointer
    for the core's attach_seq() hook.  ``model`` must be the same union
    model the core was emitted from: the union planning pass runs here only
    to reproduce the core's shared-state allocation (register indices) so
    the module folds into the same registers.
    """
    experts = experts or []
    seq_mode = seq_module_core is not None
    if arith not in ("int64", "int32", "f32"):
        raise ValueError(f"unknown arith {arith!r}")
    is_int = arith in ("int64", "int32")
    if not seq_mode and any(e.get("kind") == "seq" for e in experts):
        raise RuntimeError(
            "sequence experts are compiled per-ROM (SequenceModule) and "
            "attached at runtime, not emitted into the core engine"
        )
    if seq_mode and (
        not experts or any(e.get("kind") != "seq" for e in experts)
    ):
        raise RuntimeError("a sequence module takes kind='seq' experts only")
    if experts:
        if not is_int:
            raise RuntimeError(
                "opcode-routed expert models require an integer arith mode "
                "(f32 keeps every lookup head on a real hull, which cannot "
                "be shared across models)"
            )
        if step_spec is None:
            raise RuntimeError("opcode-routed expert models require a step_spec")

    # Shared-state prefix: "" emitting the engine itself (its own members),
    # "E." emitting a sequence module (a typed reference to the core engine).
    # Module-private scratch (xbuf, the hull_seen guards) stays unprefixed.
    E = "E." if seq_mode else ""

    d_model = model.tok.weight.shape[1]
    n_layers = len(model.attn)
    # The step_core variants share one residual buffer; an expert model may
    # schedule wider than the union (e.g. PLP), so size it for the widest.
    xbuf_size = max([d_model] + [e["model"].tok.weight.shape[1] for e in experts])

    lines: list[str] = []
    emit = lines.append

    body: list[str] = []  # per-layer member functions
    hull_tiebreak: list[int] = []  # flat hull id -> latest?
    mem_hulls: list[list[int]] = [[] for _ in range(16)]  # group -> flat hull ids
    mem_fallback_hulls: list[int] = []  # memory hulls kept on the hull path
    n_memory_heads = 0
    n_hull_heads = 0  # heads still on real hulls
    # Running-argmax registers for constant-query heads, keyed by
    # (union lookup id, value pair): the union pass allocates in head-scan
    # order, expert passes resolve through the fold's lookup provenance.
    reg_of: dict[tuple[int, int], int] = {}

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
    if seq_mode:
        layer_sig = f"Eng& E, {layer_sig}"

    def plan_model(mdl, label, union_lookup_of=None):
        """Per-layer emission plans for one model.

        The union pass (union_lookup_of is None) allocates the shared engine
        state -- hull ids, the memory-structure insert targets, and the
        running-argmax registers keyed by (lookup id, value pair) -- in
        head-scan order.  Expert passes resolve their heads onto that state
        through the fold's lookup provenance and never allocate: an expert
        head whose classification disagrees with the union's is a hard
        error, because a per-expert hull would accumulate divergent history
        across routed steps.
        """
        nonlocal n_memory_heads, n_hull_heads
        is_union = union_lookup_of is None
        dm = mdl.tok.weight.shape[1]
        nh = mdl.attn[0].num_heads
        plans: list[dict] = []
        for li in range(len(mdl.attn)):
            _check_ffn_kinds(mdl, li)
            ip = _csr(mdl.attn[li].in_proj_weight)
            op = _csr(mdl.attn[li].out_proj.weight)
            fi = _csr(mdl.ff_in[li].weight)
            fo = _csr(mdl.ff_out[li].weight)
            ffn = fo.shape[1]
            kinds = mdl.head_kinds[li]
            ffn_kinds = mdl.ffn_kinds[li]
            tiebreak = mdl.head_tiebreak[li]
            groups = mdl.head_cache_groups[li]
            mem_groups = mdl.head_memory_groups[li]
            lookup_ids = mdl.head_lookup_ids[li]
            pairs = mdl.head_pairs[li]

            lookup = [h for h in range(nh) if kinds[h] == 1]
            passthrough = [h for h in range(nh) if kinds[h] == 2]
            if is_int:
                for h in lookup:
                    if not tiebreak[h]:
                        raise RuntimeError(
                            f"{label} L{li} head {h}: AVERAGE tie-break can return "
                            "fractional hull outputs; use arith='f32'"
                        )

            # Shared hulls, first-seen in ascending head order (engine order).
            # Expert hull ids exist only for the hull_seen live-insert guard;
            # only the union's memory hulls receive insert_write tokens.
            hull_base = len(hull_tiebreak)
            hull_of_group: dict[int, int] = {}
            hull_of_head: dict[int, int] = {}
            for h in lookup:
                g = groups[h]
                if g < 0:
                    raise RuntimeError(f"{label} L{li} head {h}: negative cache group")
                if g not in hull_of_group:
                    hull_of_group[g] = hull_base + len(hull_of_group)
                    hull_tiebreak.append(int(tiebreak[h]))
                hull_of_head[h] = hull_of_group[g]
                if mem_groups[h] >= 0 and is_union:
                    mem_hulls[mem_groups[h]].append(hull_of_head[h])
                    n_memory_heads += 1

            # Structural passthrough/erase fusion: fold kind-2 heads and
            # neurons into direct D->D residual maps (no hull/ReGLU needed).
            pt_cols = [2 * h + c for h in passthrough for c in range(2)]
            attn_direct = _compose(op, pt_cols, ip[[2 * dm + c for c in pt_cols], :])
            op_kept = _zero_columns(op, pt_cols)

            lookup_cols = {2 * h + c for h in lookup for c in range(2)}
            for c in np.unique(op_kept.indices):
                if int(c) not in lookup_cols:
                    raise RuntimeError(
                        f"{label} L{li}: out_proj column {int(c)} survives at a "
                        "non-lookup head slot"
                    )

            pt_j = [j for j in range(ffn) if ffn_kinds[j] == 2]
            ffn_direct = _compose(fo, pt_j, fi[[ffn + j for j in pt_j], :])
            fo_kept = _zero_columns(fo, pt_j)
            live = sorted(int(j) for j in np.unique(fo_kept.indices))

            # Per-head emission blocks (each self-contained: it defines
            # ho{2h} and ho{2h+1}), keyed by head.
            head_blocks: dict[int, list[str]] = {}

            # ── Attention: hull heads, then residual updates ─────────
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
                    and int(q1_idx[0]) == mdl.one_slot
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
                if not is_union and mem_groups[h] >= 0 and not mem_ok:
                    raise RuntimeError(
                        f"{label} L{li} head {h}: memory head is not "
                        "structure-eligible; a per-expert hull would diverge"
                    )

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
                        len(q0_idx) == 1 and int(q0_idx[0]) == mdl.one_slot
                        and len(q1_idx) == 1 and int(q1_idx[0]) == mdl.one_slot
                    )
                    if reg_ok:
                        try:
                            v0_i = _expr_int(*_row(ip, 2 * dm + 2 * h),
                                             xref, f"{label} L{li} reg v0")
                            v1_i = _expr_int(*_row(ip, 2 * dm + 2 * h + 1),
                                             xref, f"{label} L{li} reg v1")
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
                    qi_expr = _expr_int(q_idx, q_round, xref, f"{label} L{li} mem query")
                    k1 = _expr_dbl(*_row(ip, dm + 2 * h + 1), dref)
                    b.append(f"        {elem} ho{2 * h}, ho{2 * h + 1};")
                    b.append("        {")
                    b.append(f"            if (hull_seen[{hid}] != {E}seq) {{")
                    b.append(f"                const double kyg = {k1};")
                    b.append("                if (kyg > -1.0e29)")
                    b.append('                    throw std::runtime_error('
                             '"live forward insert into a memory-group hull");')
                    b.append(f"                hull_seen[{hid}] = {E}seq;")
                    b.append("            }")
                    b.append(f"            const long long qi = shr2({qi_expr});")
                    b.append("            long long mo[2] = {0, 0};")
                    # Group dispatch: qi is the address minus this group's
                    # base, so qi in [0, 4096) iff the address falls in this
                    # group.  Other groups' contributions are gate-zeroed
                    # downstream, and (0,0) is exactly the empty-group result.
                    b.append("            if ((unsigned long long)qi >> 12 == 0)")
                    b.append(f"                {E}mem_query({g}, qi, mo);")
                    b.append(f"            ho{2 * h} = ({elem})(mo[0] << F);")
                    b.append(f"            ho{2 * h + 1} = ({elem})(mo[1] << F);")
                    b.append("        }")
                elif reg_ok:
                    # Register index: allocated by the union in scan order,
                    # resolved by experts through the fold's provenance so
                    # every model folds into / reads the same register.
                    lu_id, pair = lookup_ids[h], pairs[h]
                    if is_union:
                        key = (lu_id, pair)
                        if key in reg_of:
                            raise RuntimeError(
                                f"union L{li} head {h}: register key {key} repeated"
                            )
                        r = reg_of[key] = len(reg_of)
                    else:
                        r = reg_of.get((union_lookup_of.get(lu_id, -1), pair))
                        if r is None:
                            raise RuntimeError(
                                f"{label} L{li} head {h}: constant-query head "
                                f"has no union register (lookup {lu_id} pair {pair})"
                            )
                    qx = float(q0_dat[0])
                    qy = float(q1_dat[0])
                    k0 = _expr_dbl(*_row(ip, dm + 2 * h), dref)
                    k1 = _expr_dbl(*_row(ip, dm + 2 * h + 1), dref)
                    b.append(f"        {elem} ho{2 * h}, ho{2 * h + 1};")
                    b.append("        {")
                    b.append(f"            const double kyv = {k1};")
                    b.append("            if (kyv > -1.0e29) {  // live: fold into the running argmax")
                    b.append(f"                const double s = {_lit_d(qx)}*({k0}) + {_lit_d(qy)}*kyv;")
                    b.append(f"                if (!{E}reg_any[{r}] || s >= {E}reg_s[{r}]) {{")
                    b.append(f"                    {E}reg_any[{r}] = true;")
                    b.append(f"                    {E}reg_s[{r}] = s;")
                    b.append(f"                    {E}reg_v0[{r}] = {v0_i};")
                    b.append(f"                    {E}reg_v1[{r}] = {v1_i};")
                    b.append("                }")
                    b.append("            }")
                    b.append(f"            ho{2 * h} = {E}reg_any[{r}] ? {E}reg_v0[{r}] : 0;")
                    b.append(f"            ho{2 * h + 1} = {E}reg_any[{r}] ? {E}reg_v1[{r}] : 0;")
                    b.append("        }")
                elif is_int:
                    if not is_union:
                        raise RuntimeError(
                            f"{label} L{li} head {h}: expert lookup head needs a "
                            "real hull; refusing (its history would diverge from "
                            "the union's across routed steps)"
                        )
                    n_hull_heads += 1
                    if mem_groups[h] >= 0:
                        mem_fallback_hulls.append(hid)
                    # Q/K/V straight to double: the hull resolves in double
                    # anyway, and only these rows may read position features.
                    q0 = _expr_dbl(*_row(ip, 2 * h), dref)
                    q1 = _expr_dbl(*_row(ip, 2 * h + 1), dref)
                    k0 = _expr_dbl(*_row(ip, dm + 2 * h), dref)
                    k1 = _expr_dbl(*_row(ip, dm + 2 * h + 1), dref)
                    v0 = _expr_dbl(*_row(ip, 2 * dm + 2 * h), dref)
                    v1 = _expr_dbl(*_row(ip, 2 * dm + 2 * h + 1), dref)
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
                    k0 = _expr(*_row(ip, dm + 2 * h), xref)
                    k1 = _expr(*_row(ip, dm + 2 * h + 1), xref)
                    v0 = _expr(*_row(ip, 2 * dm + 2 * h), xref)
                    v1 = _expr(*_row(ip, 2 * dm + 2 * h + 1), xref)
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
            # These fused rows are passthrough/erase (slot-hygiene) work
            # within this model's own residual layout.
            ad_rows = [r for r in range(dm)
                       if attn_direct.indptr[r + 1] > attn_direct.indptr[r]]
            ad_def = [f"        const {elem} ad{r} = "
                      f"{row_int(attn_direct, r, xref, f'L{li} attn_direct')};"
                      for r in ad_rows]
            # out_proj rows read the hull outputs ho{..}.
            opkept: list[tuple[int, list[int], list[float]]] = []
            for r in range(dm):
                if op_kept.indptr[r + 1] > op_kept.indptr[r]:
                    idxs, dat = _row(op_kept, r)
                    opkept.append((r, [int(c) for c in idxs], list(dat)))
            ad_apply = [f"        x[{r}] += ad{r};" for r in ad_rows]

            # ── FFN: direct map and activations read post-attention x ─
            fd_rows = [r for r in range(dm)
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
            # ff_out rows read the ReGLU activations a{..}.
            fokept: list[tuple[int, list[int], list[float]]] = []
            for r in range(dm):
                if fo_kept.indptr[r + 1] > fo_kept.indptr[r]:
                    idxs, dat = _row(fo_kept, r)
                    fokept.append((r, [int(c) for c in idxs], list(dat)))
            fd_apply = [f"        x[{r}] += fd{r};" for r in fd_rows]

            plans.append({
                "li": li, "lookup": lookup, "head_blocks": head_blocks,
                "ad_def": ad_def, "opkept": opkept, "ad_apply": ad_apply,
                "fd_def": fd_def, "live": live, "neuron_blocks": neuron_blocks,
                "fokept": fokept, "fd_apply": fd_apply,
            })
        return plans

    layer_plans = plan_model(model, "union")
    expert_plans = [
        (exp, plan_model(exp["model"], f"expert {exp['name']}",
                         exp["union_lookup_of"]))
        for exp in experts
    ]
    n_regs = len(reg_of)

    # ── Emit the layer functions: one body per model -- the union stack
    # plus each rebuilt expert's own (smaller) stack.  Expert bodies are
    # complete emissions of their own weight matrices, not pruned copies of
    # the union's; the fold already removed the dead work at graph level. ──
    def emit_layer_fn(plan, suffix):
        li = plan["li"]
        out = [f"    void layer_{li}{suffix}({layer_sig}) {{"]
        for h in plan["lookup"]:
            out.extend(plan["head_blocks"][h])
        out.extend(plan["ad_def"])
        for r, idxs, dat in plan["opkept"]:
            out.append(f"        x[{r}] += "
                       f"{row_raw(idxs, dat, lambda i: f'ho{i}', f'L{li} out_proj')};")
        out.extend(plan["ad_apply"])
        out.extend(plan["fd_def"])
        for j in plan["live"]:
            out.extend(plan["neuron_blocks"][j])
        for r, idxs, dat in plan["fokept"]:
            out.append(f"        x[{r}] += "
                       f"{row_raw(idxs, dat, lambda i: f'a{i}', f'L{li} ff_out')};")
        out.extend(plan["fd_apply"])
        out.append("    }")
        return out

    # In seq mode the union plans exist only to reproduce the core's
    # shared-state allocation; the module emits expert bodies alone.
    if not seq_mode:
        for plan in layer_plans:
            body.extend(emit_layer_fn(plan, ""))
    for exp, plans in expert_plans:
        for plan in plans:
            body.extend(emit_layer_fn(plan, f"__{exp['name']}"))

    # ── Step-core emission helpers, shared by the core engine and the
    # per-ROM sequence module ────────────────────────────────────────
    if is_int:
        one_e, zero_l = "SCALE", "0"

        def sc(v, e=elem):
            return f"(({e})({v}) << F)"
    else:
        one_e, zero_l = "1.0f", "0.0f"

        def sc(v):
            return f"(float)({v})"

    rdv = (lambda s: f"shr2(x[{s}])") if is_int else (lambda s: f"rd(x[{s}])")

    def emit_step_core(fn_name, layer_suffix, mdl, spec):
        # One whole instruction, sharing the embed/readout/write/halt
        # protocol; layer_suffix selects the union stack ("") or a routed
        # expert's own rebuilt stack ("__<name>").  Each model embeds and
        # reads out through its own slot layout (spec); the persistent
        # st_* registers are the cross-model interface.
        ins = spec["inputs"]
        # A sequence expert reads out K instructions from one forward
        # (per-step slot maps under "step_outputs"); a plain expert is
        # the K=1 case.
        step_outs = spec.get("step_outputs") or [spec["outputs"]]
        hc, hv = spec["halt_head"]
        shc, shv = spec["state_head"]
        dm = mdl.tok.weight.shape[1]

        emit(f"    void {fn_name}({'Eng& E, ' if seq_mode else ''}long long* out) {{")
        emit(f"        {elem}* x = xbuf;")
        emit(f"        for (int i = 0; i < {dm}; ++i) x[i] = {zero};")
        emit(f"        x[{mdl.one_slot}] = {one_e};")
        emit(f"        x[{ins['is_state']}] = {one_e};")
        emit(f"        x[{ins['pc_lo']}] = {sc(f'{E}st_pc & 0xFF')};")
        emit(f"        x[{ins['pc_hi']}] = {sc(f'({E}st_pc >> 8) & 0xFF')};")
        emit(f"        x[{ins['a']}] = {sc(f'{E}st_a')};")
        emit(f"        x[{ins['x']}] = {sc(f'{E}st_x')};")
        emit(f"        x[{ins['y']}] = {sc(f'{E}st_y')};")
        emit(f"        x[{ins['s']}] = {sc(f'{E}st_s')};")
        for slot, mask in spec["p_flags"]:
            emit(f"        x[{slot}] = ({E}st_p & {mask}) ? {one_e} : {zero_l};")
        emit(f"        x[{ins['ir']}] = {sc(f'{E}st_ir')};")
        emit(f"        x[{ins['oplo']}] = {sc(f'{E}st_oplo')};")
        emit(f"        x[{ins['ophi']}] = {sc(f'{E}st_ophi')};")
        emit(f"        x[{ins['ea_lo']}] = {sc(f'{E}st_ea & 0xFF')};")
        emit(f"        x[{ins['ea_hi']}] = {sc(f'({E}st_ea >> 8) & 0xFF')};")
        emit(f"        x[{ins['data']}] = {sc(f'{E}st_data')};")
        emit(f"        x[{ins['halted']}] = {sc(f'{E}st_halted ? 1 : 0')};")
        emit(f"        x[{ins['nmi']}] = {sc(f'{E}nmi_pending ? 1 : 0')};")
        emit(f"        {E}nmi_pending = false;")
        emit(f"        const long long pos = {E}seq + 1;")
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
        emit(f"        ++{E}seq;")
        for li in range(len(mdl.attn)):
            emit(f"        layer_{li}{layer_suffix}("
                 f"{'E, ' if seq_mode else ''}x{', pf' if is_int else ''});")
        for k, outs in enumerate(step_outs):
            off = 21 * k
            for i, (name, mask) in enumerate((
                ("current_pc", 0xFFFF), ("new_a", 0xFF), ("new_x", 0xFF),
                ("new_y", 0xFF), ("new_s", 0xFF), ("new_p", 0xFF),
                ("fetched_opcode", 0xFF), ("fetched_operand_lo", 0xFF),
                ("fetched_operand_hi", 0xFF), ("ea", 0xFFFF), ("data_out", 0xFF),
                ("new_halted", 0xFF), ("next_pc", 0xFFFF),
            )):
                emit(f"        out[{off + i}] = {rdv(outs[name])} & 0x{mask:X};")
            emit("        { long long n_writes = 0;")
            for store, addr, value in (
                ("is_store", "write_addr_out", "write_value_out"),
                ("is_store2", "write_addr2_out", "write_value2_out"),
                ("is_store3", "write_addr3_out", "write_value3_out"),
            ):
                emit(f"        if (({rdv(outs[store])} & 0xFF) != 0) {{")
                emit(f"            out[{off + 15} + 2 * n_writes] = {rdv(outs[addr])} & 0xFFFF;")
                emit(f"            out[{off + 16} + 2 * n_writes] = {rdv(outs[value])} & 0xFF;")
                emit("            ++n_writes;")
                emit("        }")
            emit(f"        out[{off + 14}] = n_writes;")
            emit("        for (long long i = 0; i < n_writes; ++i)")
            emit(f"            {E}insert_write_f32((int)out[{off + 15} + 2 * i], "
                 f"(int)out[{off + 16} + 2 * i], {E}seq + 1);")
            emit("        }")
            if k + 1 < len(step_outs):
                # Non-final rows of a sequence are generation-verified
                # non-halting; only the final row competes STATE vs HALT.
                emit(f"        out[{off + 13}] = 0;")
        last = 21 * (len(step_outs) - 1)
        if is_int:
            emit(f"        const long long halt_logit = "
                 f"{_expr_int(hc, hv, lambda i: f'x[{i}]', 'halt head')};")
            emit(f"        const long long state_logit = "
                 f"{_expr_int(shc, shv, lambda i: f'x[{i}]', 'state head')};")
        else:
            emit(f"        const float halt_logit = {_expr(hc, hv, lambda i: f'x[{i}]')};")
            emit(f"        const float state_logit = {_expr(shc, shv, lambda i: f'x[{i}]')};")
        emit(f"        out[{last + 13}] = halt_logit > state_logit ? 1 : 0;")
        emit("        // Advance the architectural state (inputs are fully")
        emit("        // consumed by the embed above before this update).")
        emit(f"        {E}st_pc = (int)out[{last + 12}];")
        emit(f"        {E}st_a = (int)out[{last + 1}]; {E}st_x = (int)out[{last + 2}]; "
             f"{E}st_y = (int)out[{last + 3}];")
        emit(f"        {E}st_s = (int)out[{last + 4}]; {E}st_p = (int)out[{last + 5}];")
        emit(f"        {E}st_ir = (int)out[{last + 6}]; {E}st_oplo = (int)out[{last + 7}]; "
             f"{E}st_ophi = (int)out[{last + 8}];")
        emit(f"        {E}st_ea = (int)out[{last + 9}]; {E}st_data = (int)out[{last + 10}];")
        emit(f"        {E}st_halted = out[{last + 11}] != 0;")
        emit("    }")

    if seq_mode:
        return _assemble_seq_module(
            lines, body, experts, seq_module_core, elem,
            len(hull_tiebreak), emit_step_core,
        )

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
        emit(f"    {elem} xbuf[{xbuf_size}];")
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
        emit("    // Read lease: watched reads of this address are served natively")
        emit("    // with this value (device semantics frozen until the bus clears it).")
        emit("    int lease_addr = -1, lease_value = 0;")
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
        emit("        std::memset(shape_of_pc, 0, sizeof(shape_of_pc));")
        emit("        std::memset(fallback_count, 0, sizeof(fallback_count));")
        emit("        st_pc = 0; st_a = st_x = st_y = 0; st_s = 0xFD; st_p = 0x24;")
        emit("        st_ir = st_oplo = st_ophi = st_ea = st_data = 0;")
        emit("        st_halted = false; nmi_pending = false; resume_read_skip = false;")
        emit("        lease_addr = -1;")
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
        emit("        const float ky_f = -local * local;")
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
        emit_step_core("step_core_union", "", model, step_spec)
        for exp in experts:
            emit("")
            emit_step_core(f"step_core__{exp['name']}", f"__{exp['name']}",
                           exp["model"], exp["step_spec"])

        # Opcode-routed dispatch (the MoE router): the fetched opcode is
        # ram[st_pc] (== the graph's memory read at pc), so an expert built by
        # pinning is_op_<byte> runs exactly when that byte is fetched.  A
        # pending NMI overrides the instruction, so it takes the union step.
        op_experts = [e for e in experts if e.get("kind") is None]
        shape_experts = [e for e in experts if e.get("kind") == "shape"]
        emit("")
        emit("    void step_core(long long* out) {")
        if op_experts:
            emit("        if (!nmi_pending) {")
            emit("            switch (ram[st_pc]) {")
            for exp in op_experts:
                emit(f"                case 0x{exp['byte']:02X}: "
                     f"step_core__{exp['name']}(out); return;")
            emit("            }")
            emit("        }")
        emit("        step_core_union(out);")
        emit("    }")

        # Canonical shape windows (native run loop only): compiled once into
        # this core with pc and operands symbolic, dispatched through a
        # runtime table -- zero per-ROM compile.  add_window() entries come
        # from the boot-time image scan; dispatch re-verifies the shape's
        # opcode bytes (and the JMP self-target) against live RAM, so a
        # stale entry only loses the fast path, never correctness.
        emit("")
        emit("    unsigned char shape_of_pc[65536];")
        n_shapes = len(shape_experts)
        if n_shapes:
            lens = ", ".join(str(e["length"]) for e in shape_experts)
            emit(f"    static constexpr int SHAPE_LEN[{n_shapes}] = {{ {lens} }};")
        emit("    void add_window(int shape, int head) {")
        if n_shapes:
            emit(f"        if (shape < 0 || shape >= {n_shapes})")
            emit('            throw std::runtime_error("unknown shape id");')
            emit("        if (head < 0 || head + SHAPE_LEN[shape] > 0x10000)")
            emit('            throw std::runtime_error("shape window would wrap $FFFF");')
            emit("        shape_of_pc[head] = (unsigned char)(shape + 1);")
        else:
            emit("        (void)shape; (void)head;")
            emit('        throw std::runtime_error("engine has no shape experts");')
        emit("    }")
        emit("    void clear_windows() { std::memset(shape_of_pc, 0, sizeof(shape_of_pc)); }")
        emit("")
        emit("    int shape_dispatch(long i, long max_steps, long cycles_this,")
        emit("                       long max_cycles, bool stop_on_trap) {")
        if n_shapes:
            emit("        const int sid = shape_of_pc[st_pc];")
            emit("        if (!sid || nmi_pending || stop_on_trap) return 0;")
            emit("        switch (sid) {")
            for j, exp in enumerate(shape_experts):
                # add_window guarantees head + length <= 0x10000: no wrap.
                checks = " && ".join(
                    f"ram[st_pc + {off}] == 0x{b:02X}" if off
                    else f"ram[st_pc] == 0x{b:02X}"
                    for off, b in exp["opcode_checks"]
                )
                if exp.get("self_target"):
                    checks += (" && ram[st_pc + 1] == (st_pc & 0xFF)"
                               " && ram[st_pc + 2] == ((st_pc >> 8) & 0xFF)")
                emit(f"        case {j + 1}:  // {exp['name']}")
                emit(f"            if (i + {exp['n_steps']} <= max_steps")
                emit(f"                && (max_cycles <= 0 || cycles_this + "
                     f"{exp['cycles_prefix']} < max_cycles)")
                emit(f"                && {checks}) {{")
                emit(f"                step_core__{exp['name']}(seq_out);")
                emit(f"                return {exp['n_steps']};")
                emit("            }")
                emit("            return 0;")
            emit("        }")
        emit("        return 0;")
        emit("    }")

        # Sequence dispatch (native run loop only): a whole pinned window as
        # one forward, compiled per-ROM into a separate module that attaches
        # through the function-pointer hook -- no core recompile per ROM.
        emit("")
        emit(_SEQ_HOOK)
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
                 arith: str = "int32", experts=None, cache_key: str | None = None,
                 shapes_meta=None):
        self.arith = arith
        self.n_experts = len(experts or [])
        # Shape-window scan metadata (id/signature/length/close rule) so a
        # cache-hit load can populate the runtime table without refolding.
        self.shapes_meta = list(shapes_meta or [])
        source, self.n_memory_heads, self.has_machine_step = generate_source(
            model, step_spec, arith, experts=experts
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.digest = digest
        source = source.replace("ENGINE_T", f"Engine_{digest}")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        hpp_path = os.path.join(_CACHE_DIR, f"engine_{digest}.hpp")
        if not os.path.exists(hpp_path):
            n_shards = max(1, min((os.cpu_count() or 4) - 2, 12))
            header, shards = _shard_source(source, f"Engine_{digest}", n_shards)
            for k, body in enumerate(shards):
                with open(os.path.join(_CACHE_DIR, f"engine_{digest}_s{k}.cpp"),
                          "w") as f:
                    f.write(f'#include "engine_{digest}.hpp"\n\n{body}')
            with open(hpp_path, "w") as f:
                f.write(header)
        module, self.compile_seconds = _load_engine_module(digest, arith)
        if cache_key is not None:
            _manifest_put(cache_key, {
                "digest": digest, "arith": arith, "n_experts": self.n_experts,
                "n_memory_heads": self.n_memory_heads,
                "has_machine_step": self.has_machine_step,
                "shapes": self.shapes_meta,
            })
        self._bind(module)

    @classmethod
    def from_cache(cls, cache_key: str) -> GeneratedWeightsEngine | None:
        """Load an engine by config key alone, skipping the expert folds and
        the source emission entirely; None on any miss."""
        entry = _manifest_read().get(cache_key)
        if entry is None:
            return None
        if not os.path.exists(
            os.path.join(_CACHE_DIR, f"engine_{entry['digest']}.hpp")
        ):
            return None
        self = cls.__new__(cls)
        self.arith = entry["arith"]
        self.digest = entry["digest"]
        self.n_experts = entry["n_experts"]
        self.n_memory_heads = entry["n_memory_heads"]
        self.has_machine_step = entry["has_machine_step"]
        self.shapes_meta = entry.get("shapes") or []
        module, self.compile_seconds = _load_engine_module(
            entry["digest"], entry["arith"]
        )
        self._bind(module)
        return self

    def _bind(self, module) -> None:
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
            self.set_read_lease = self._engine.set_read_lease
            self.clear_read_lease = self._engine.clear_read_lease
            self.set_nmi = self._engine.set_nmi
            self.engine_inject_write = self._engine.inject_write
            self.read_ram = self._engine.read_ram
            self.dump_ram = self._engine.dump_ram
            self.attach_seq = self._engine.attach_seq
            self.detach_seq = self._engine.detach_seq
            self.add_window = self._engine.add_window
            self.clear_windows = self._engine.clear_windows
            self.jit_hot = self._engine.jit_hot
            self.window_shape = self._engine.window_shape

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


class SequenceModule:
    """Per-ROM sequence-window module: the folded window experts compiled
    against a core engine's generated header (a single small TU, seconds to
    build, cached by digest), exporting its dispatch as a raw function
    pointer for the core's attach_seq() hook.  The holder must keep this
    object alive while it is attached."""

    def __init__(self, model, step_spec, experts, *, core_digest: str,
                 arith: str, cache_key: str | None = None):
        source, _, _ = generate_source(
            model, step_spec, arith, experts=experts,
            seq_module_core=core_digest,
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        source = source.replace("SEQMOD_T", f"SeqMod_{digest}")
        self.core_digest = core_digest
        self.digest = digest
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, f"seqmod_{digest}.cpp")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(source)
        module, self.compile_seconds = _load_seqmod_module(digest, arith)
        if cache_key is not None:
            _manifest_put(cache_key, {
                "kind": "seqmod", "digest": digest, "arith": arith,
                "core": core_digest, "n_windows": len(experts),
            })
        self._bind(module)

    @classmethod
    def from_cache(cls, cache_key: str, *, core_digest: str) -> SequenceModule | None:
        """Load a module by config key alone, skipping the sequence folds
        and the emission entirely; None on any miss."""
        entry = _manifest_read().get(cache_key)
        if (
            entry is None
            or entry.get("kind") != "seqmod"
            or entry["core"] != core_digest
            or not os.path.exists(
                os.path.join(_CACHE_DIR, f"seqmod_{entry['digest']}.cpp")
            )
        ):
            return None
        self = cls.__new__(cls)
        self.core_digest = core_digest
        self.digest = entry["digest"]
        module, self.compile_seconds = _load_seqmod_module(
            entry["digest"], entry["arith"]
        )
        self._bind(module)
        return self

    def _bind(self, module) -> None:
        if module.core_name != f"Engine_{self.core_digest}":
            raise RuntimeError(
                f"sequence module was compiled against {module.core_name}, "
                f"not Engine_{self.core_digest}"
            )
        self.max_k = int(module.max_k)
        self._mod = module.SeqModule()
        self.fn_ptr = int(module.SeqModule.fn_ptr())
        self.ctx_ptr = int(self._mod.ctx_ptr())
