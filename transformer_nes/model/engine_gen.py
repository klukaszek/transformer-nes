"""Emit model-specific C++ for analytic transformer weights.

Sparse indices and coefficients become C++ expressions, removing CSR gather
overhead. The residual uses exact fixed-point int32. Attention uses exact
memory structures, running argmax registers, or hulls according to
:mod:`model.classify`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from transformer_nes.cpu6502.backends.weights.run_protocol import (
    IN_RANGES_CPP,
    MACHINE_API_CPP,
    SHADOW_READ_TARGET_CPP,
    mem_query_cpp,
    run_loop_cpp,
    run_state_members_cpp,
)
from transformer_nes.graph.core import MEMORY_GROUP_FUSED
from transformer_nes.model.classify import (
    integer_score_terms,
    memory_query,
    register_query,
)

ATTENTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention"
)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_engine_cache")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
# Shards bound the size of each generated C++ translation unit; jobs bound how
# many compiler frontends are resident at once.  They are deliberately
# independent: a few giant shards still make one clang process consume tens of
# gigabytes, while many unconstrained shards multiply the smaller peak.
DEFAULT_COMPILE_SHARDS = 12
DEFAULT_COMPILE_JOBS = 2


def positive_env_int(name: str, default: int) -> int:
    """Read a positive integer build setting without leaking stringly typed
    environment handling through the engine/cache paths."""
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def compile_shards() -> int:
    return positive_env_int("TRANSFORMER_NES_COMPILE_SHARDS", DEFAULT_COMPILE_SHARDS)

# Everything whose content can change the emitted engine, hashed into the
# config key so any edit is a cache miss (over-inclusion is safe).
CODE_DIRS = ("graph", "scheduler", "model", "attention")
CODE_FILES = (
    "cpu6502/spec.py",
    "cpu6502/interpreter.py",
    "cpu6502/backends/graph.py",
    "cpu6502/backends/reference.py",
    "cpu6502/backends/weights/run_protocol.py",
    "cpu6502/backends/weights/windows.py",
    "cpu6502/backends/weights/experts.py",
    "cpu6502/backends/weights/runner.py",
)
code_digest_cache: str | None = None


def code_digest() -> str:
    global code_digest_cache
    if code_digest_cache is not None:
        return code_digest_cache
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = []
    for d in CODE_DIRS:
        for root, dirs, files in os.walk(os.path.join(pkg, d)):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "_engine_cache")]
            paths.extend(
                os.path.join(root, f) for f in files
                if f.endswith((".py", ".h", ".hpp", ".cpp"))
            )
    paths.extend(os.path.join(pkg, *f.split("/")) for f in CODE_FILES)
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: os.path.relpath(p, pkg)):
        h.update(os.path.relpath(p, pkg).encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    code_digest_cache = h.hexdigest()
    return code_digest_cache


def engine_config_key(*, bcd: bool, expert_bytes, extra: str = "") -> str:
    """Cache key for a built engine, computable without folding or emitting:
    the config plus a digest of every source file that shapes the emission."""
    tag = ",".join(f"{int(b):02X}" for b in expert_bytes)
    raw = f"{code_digest()}|bcd={int(bcd)}|experts={tag}|{extra}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def manifest_read() -> dict:
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def manifest_put(key: str, entry: dict) -> None:
    manifest = manifest_read()
    manifest[key] = entry
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST_PATH)


FUNC_RE = re.compile(r"^    void ((?:layer_|step_core)\w*)\((.*)\) \{$")


def shard_source(source: str, struct_name: str, n_shards: int) -> tuple[str, list[str]]:
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
        m = FUNC_RE.match(line)
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

    shards: list[list[str]] = [[] for _ in range(max(1, n_shards))]
    sizes = [0] * len(shards)
    for chunk in sorted(chunks, key=lambda c: len(c), reverse=True):
        k = sizes.index(min(sizes))
        shards[k].extend(chunk)
        shards[k].append("")
        sizes[k] += len(chunk)
    shards[0] = pybind + [""] + shards[0]
    return "\n".join(header) + "\n", ["\n".join(s) + "\n" for s in shards]


def load_module(
    name: str, digest: str, sources: list[str]
) -> tuple[object, float]:
    """Load a cached extension or compile it from ``sources``."""
    if name in sys.modules:
        return sys.modules[name], 0.0
    build_dir = os.path.join(CACHE_DIR, f"build_{digest}")
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.exists(so_path):
        spec = importlib.util.spec_from_file_location(name, so_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module, 0.0

    from torch.utils.cpp_extension import load

    build_start = time.perf_counter()
    cflags = ["-O2", "-std=c++17", "-fwrapv"]
    os.makedirs(build_dir, exist_ok=True)
    old_max_jobs = os.environ.get("MAX_JOBS")
    if old_max_jobs is None:
        os.environ["MAX_JOBS"] = str(
            positive_env_int(
                "TRANSFORMER_NES_COMPILE_JOBS", DEFAULT_COMPILE_JOBS
            )
        )
    try:
        module = load(
            name=name,
            sources=sources,
            extra_cflags=cflags,
            extra_include_paths=[ATTENTION_DIR, CACHE_DIR],
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        if old_max_jobs is None:
            del os.environ["MAX_JOBS"]
    return module, time.perf_counter() - build_start


def load_engine_module(digest: str) -> tuple[object, float]:
    shard_glob = sorted(
        f for f in os.listdir(CACHE_DIR)
        if f.startswith(f"engine_{digest}_s") and f.endswith(".cpp")
    )
    sources = [os.path.join(CACHE_DIR, path) for path in shard_glob]
    return load_module(f"weights_gen_{digest}", digest, sources)


def seqmod_config_key(*, core_digest: str, bcd: bool, windows) -> str:
    """Cache key for a per-ROM sequence module.  The core digest covers the
    header the module compiles against; the code digest covers fold/emission
    changes that alter the module without altering the core's source."""
    tag = ",".join(f"{h:04X}:{bytes(c).hex()}x{r}" for h, c, r in windows)
    raw = f"{code_digest()}|core={core_digest}|bcd={int(bcd)}|seq={tag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_seqmod_module(digest: str) -> tuple[object, float]:
    return load_module(
        f"seqmod_gen_{digest}",
        digest,
        [os.path.join(CACHE_DIR, f"seqmod_{digest}.cpp")],
    )


def lit_d(value: float) -> str:
    """A float64 value as an exact C++ hexfloat literal."""
    f = float(value)
    if f == int(f) and abs(f) < 2**53:
        return f"{int(f)}.0"
    return f.hex()


def join(terms) -> str:
    if not terms:
        return ""
    sign, first = terms[0]
    out = f"-{first}" if sign == "-" else first
    for sign, term in terms[1:]:
        out += f" {sign} {term}"
    return out


def expr_dbl(indices, data, ref) -> str:
    """One CSR row as a float64 expression (hull-bound Q/K/V rows)."""
    terms = []
    for idx, c in zip(indices, data, strict=True):
        c = float(c)
        if c == 1.0:
            terms.append(("+", ref(idx)))
        elif c == -1.0:
            terms.append(("-", ref(idx)))
        else:
            terms.append(("+", f"{lit_d(c)}*{ref(idx)}"))
    return join(terms) or "0.0"


def expr_int(indices, data, ref, where: str) -> str:
    """One CSR row as an exact fixed-point integer expression.

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
                "exact int32 emission"
            )
    return join(terms) or "0"


def row(csr, r) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = csr.indptr[r], csr.indptr[r + 1]
    return csr.indices[lo:hi], csr.data[lo:hi]


def csr(array) -> sp.csr_matrix:
    m = sp.csr_matrix(array)
    m.sort_indices()
    return m


def zero_columns(m: sp.csr_matrix, cols) -> sp.csr_matrix:
    out = m.tolil()
    for c in cols:
        out[:, c] = 0
    out = out.tocsr()
    out.eliminate_zeros()
    out.sort_indices()
    return out


def compose(outer: sp.csr_matrix, cols, inner_rows: sp.csr_matrix) -> sp.csr_matrix:
    """outer[:, cols] @ inner_rows, like the engine's direct-map fusion.

    Duplicate contributions merge in float64 and exact zeros are dropped,
    matching csr_from_triplets.
    """
    direct = (outer.tocsc()[:, list(cols)] @ inner_rows).tocsr()
    direct.eliminate_zeros()
    direct.sort_indices()
    return direct


def fuse_direct(mat, rows, var, pre_written, label, *, elem, xref):
    """Direct-map emission with the hygiene peephole: a row whose operands
    are untouched by the phase's earlier writes applies as ONE statement
    (an erase with an unwritten target becomes assignment); anything else
    keeps the capture+apply pair.  Returns (defs, applies)."""
    defs, applies = [], []
    written = set(pre_written)
    for r in rows:
        idxs, dat = row(mat, r)
        terms = list(zip(idxs, dat, strict=True))
        diag = dict(terms).get(r)
        rest = [(i, c) for i, c in terms if i != r]
        support = {int(i) for i, _ in rest}
        if diag == -1.0 and r not in written and not support & written:
            if rest:
                e = expr_int([i for i, _ in rest],
                             [c for _, c in rest],
                             xref, f"{label} fused")
                applies.append(f"        x[{r}] = {e};")
            else:
                applies.append(f"        x[{r}] = 0;")
        elif diag is None and not support & written:
            e = expr_int(idxs, dat, xref, f"{label} fused")
            applies.append(f"        x[{r}] += {e};")
        else:
            defs.append(f"        const {elem} {var}{r} = "
                        f"{expr_int(*row(mat, r), xref, label)};")
            applies.append(f"        x[{r}] += {var}{r};")
        written.add(int(r))
    return defs, applies


def check_ffn_kinds(model, layer_idx):
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

INCLUDES = """\
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
FIXED_POINT_HELPERS = """\
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

# Exact nearest-written-address argmax over one 4 KiB memory group; the
# emission comes from run_protocol.py.
MEM_QUERY = mem_query_cpp("long long")

# Per-ROM sequence-module hook: a separately compiled window module attaches
# its dispatch through a raw function pointer, so a new ROM's window set never
# recompiles this TU.  Execution models return one final state and apply their
# ordered writes directly; traced runs bypass them.  A pinned module window
# outranks the canonical shape table (it folded the operands too, so its
# forward is cheaper); both fall back to per-step.
SEQ_HOOK = """\
    typedef int (*SeqFn)(void*, void*, long, long, long, long, int);
    SeqFn seq_fn = nullptr;
    void* seq_ctx = nullptr;
    static constexpr int SEQ_MAX_K = 32;
    long seq_cycles = 0;
    // Module window heads: dispatch misses cost one array load instead
    // of a cross-module call into the module's head switch.
    unsigned char seq_head[65536] = {};

    int seq_dispatch(long i, long max_steps, long cycles_this,
                     long max_cycles, bool stop_on_trap) {
        if (seq_fn && seq_head[st_pc]) {
            const int k = seq_fn(seq_ctx, this, i, max_steps, cycles_this,
                                 max_cycles, stop_on_trap ? 1 : 0);
            if (k) return k;
        }
        const int k = shape_dispatch(i, max_steps, cycles_this, max_cycles,
                                     stop_on_trap);
        return k;
    }

    void attach_seq(std::uintptr_t fn, std::uintptr_t ctx, py::bytes heads) {
        seq_fn = (SeqFn)fn; seq_ctx = (void*)ctx;
        std::memset(seq_head, 0, sizeof(seq_head));
        const std::string h = heads;
        for (size_t i = 0; i + 1 < h.size(); i += 2) {
            const int pc = (unsigned char)h[i] | ((unsigned char)h[i + 1] << 8);
            seq_head[pc] = 1;
        }
    }
    void detach_seq() {
        seq_fn = nullptr; seq_ctx = nullptr;
        std::memset(seq_head, 0, sizeof(seq_head));
    }"""

# Per-iteration body of the shared run loop (run_protocol.run_loop_cpp):
# sequence dispatch, then the per-instruction step_core fallback.
ENGINE_STEP_BODY = """\
            // A routed sequence consumes a whole window in one forward;
            // its guards (in seq_dispatch) keep this step-for-step
            // equivalent to the per-instruction path below.
            if (!trace) {
                const int k_seq = seq_dispatch(i, max_steps, cycles_this,
                                               max_cycles, stop_on_trap);
                if (k_seq > 0) {
                    step_index += k_seq;
                    cycles_this += seq_cycles;
                    cycles_total += seq_cycles;
                    if (st_halted) {
                        st_halted = true; reason = "halt"; break;
                    }
                    i += k_seq - 1;
                    if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }
                    continue;
                }
            }
            const bool was_int = nmi_pending || irq_ready();
            const int prev_pc = st_pc;
            ++fallback_count[prev_pc];  // JIT profile: per-instruction steps only
            step_core(o);  // updates st_*, applies writes to ram + memory
            if (trace) {
                t_pc.push_back((int)o[0]); t_a.push_back((int)o[1]);
                t_x.push_back((int)o[2]); t_y.push_back((int)o[3]);
                t_s.push_back((int)o[4]); t_p.push_back((int)o[5]);
                t_ir.push_back((int)o[6]); t_oplo.push_back((int)o[7]);
                t_ophi.push_back((int)o[8]); t_ea.push_back((int)o[9]);
                t_data.push_back((int)o[10]);
            }
            const bool halted = o[11] != 0;
            if (trace) t_halted.push_back(halted ? 1 : 0);
            const int n_writes = (int)o[14];
            int n_hits = 0, sh_addr[3], sh_value[3];
            for (int w = 0; w < n_writes; ++w) {
                const int wa = (int)o[15 + 2 * w], wv = (int)o[16 + 2 * w];
                if (trace) {
                    w_row.push_back(step_index); w_addr.push_back(wa); w_value.push_back(wv);
                }
                if (wa >= 0x8000 && mapper_kind) {
                    mapper_write(wa, wv);  // native banking; no pause
                } else if (wjournal[wa]) {
                    journal_write(wa, wv);  // order-only device state; no pause
                } else if (!wwatch.empty() && in_ranges(wwatch, wa)) {
                    sh_addr[n_hits] = wa; sh_value[n_hits] = wv; ++n_hits;
                }
            }
            ++step_index;
            const long step_cycles = was_int ? 7 : (long)CYCLES[(unsigned char)o[6]];
            cycles_this += step_cycles; cycles_total += step_cycles;
            for (int k = 0; k < n_hits; ++k) { hit_addr.push_back(sh_addr[k]); hit_value.push_back(sh_value[k]); }
            if (halted) { st_halted = true; reason = "halt"; break; }
            if (o[13]) { st_halted = true; reason = "halt"; break; }
            if (stop_on_trap && (int)o[12] == prev_pc) { reason = "trap"; break; }
            if (n_hits > 0) { reason = "watch_write"; break; }
            if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }"""

# Shadow-decode + the shared run loop + machine API (run_protocol.py blocks;
# the run-loop protocol blocks).
RUN_LOOP = "\n\n".join((
    SHADOW_READ_TARGET_CPP,
    IN_RANGES_CPP,
    run_loop_cpp(extra_locals="        long long o[21];",
                 step_body=ENGINE_STEP_BODY),
    """\
    // ── JIT profile counter ──────────────────────────────────
    // The run loop counts every per-instruction (fallback) step by pc;
    // jit_hot reports pcs past the cumulative threshold.  Canonical shape
    // dispatches count too: hot symbolic shapes are candidates for cheaper
    // operand-pinned per-ROM promotion on the next boot.
    unsigned int fallback_count[65536];

    py::array_t<int> jit_hot(long min_count) {
        std::vector<int> hot;
        for (int pc = 0; pc < 65536; ++pc)
            if ((long)fallback_count[pc] >= min_count) {
                hot.push_back(pc);
                fallback_count[pc] = 0;
            }
        return py::array_t<int>((py::ssize_t)hot.size(), hot.data());
    }

    int window_shape(int pc) { return shape_of_pc[pc & 0xFFFF]; }

    // ── Machine API (run_protocol.py) ────────────────────────""",
    MACHINE_API_CPP,
    """\
    void inject_write(int addr, int value) { insert_write(addr & 0xFFFF, value & 0xFF, seq + 1); }

    // Cartridge mapping changes address translation; it does not append
    // thousands of fictional CPU WRITE tokens.  All 6502 memory heads use
    // this exact-address table, so replacing a mapped byte in place preserves
    // reads while leaving transformer sequence position unchanged.
    void map_write(int addr, int value) {
        addr &= 0xFFFF; value &= 0xFF;
        ram[addr] = (unsigned char)value;
        const int group = addr >> 12, wl = addr & 0xFFF;
        mem_val[group][wl] = value;
        mem_seq[group][wl] = seq;
        mem_bits[group][wl >> 6] |= 1ULL << (wl & 63);
    }

    void map_region(int base, py::bytes data) {
        const std::string d = data;
        for (size_t i = 0; i < d.size(); ++i)
            map_write(base + (int)i, (unsigned char)d[i]);
    }

    // Bulk injection (mapper bank switches): one call per region instead
    // of one Python call per byte.
    void inject_region(int base, py::bytes data) {
        const std::string d = data;
        for (size_t i = 0; i < d.size(); ++i)
            insert_write((base + (int)i) & 0xFFFF, (unsigned char)d[i], seq + 1);
    }

    // Boot/reseed seed: the whole image in one call, nonzero bytes only
    // (matching boot_image's Python-side semantics); returns tokens added.
    long seed_image(py::bytes image) {
        const std::string d = image;
        if (d.size() != 65536)
            throw std::runtime_error("seed_image needs a 64 KiB image");
        long n = 0;
        for (int a = 0; a < 65536; ++a) {
            const unsigned char v = (unsigned char)d[a];
            if (v) { insert_write(a, v, seq + 1); ++n; }
        }
        return n;
    }

    void set_run_state(int pc, int a, int x, int y, int s, int p,
                       int ir, int oplo, int ophi, int ea, int data, int halted) {
        st_pc = pc & 0xFFFF; st_a = a & 0xFF; st_x = x & 0xFF; st_y = y & 0xFF;
        st_s = s & 0xFF; st_p = (p | 0x20) & 0xFF;
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
    }""",
))

# The pybind .def() lines gated on has_step (the machine_step + run-loop API).
PYBIND_STEP_METHODS = """\
        .def("machine_step", &ENGINE_T::machine_step)
        .def("run", &ENGINE_T::run, py::arg("max_steps"),
             py::arg("stop_on_trap") = false, py::arg("max_cycles") = 0L,
             py::arg("trace") = true)
        .def("watch_read", &ENGINE_T::watch_read)
        .def("watch_write", &ENGINE_T::watch_write)
        .def("clear_watches", &ENGINE_T::clear_watches)
        .def("set_write_journal", &ENGINE_T::set_write_journal)
        .def("set_read_lease", &ENGINE_T::set_read_lease,
             py::arg("addr"), py::arg("value"), py::arg("group") = 0)
        .def("set_read_lease_seq", &ENGINE_T::set_read_lease_seq,
             py::arg("addr"), py::arg("values"), py::arg("group") = 0)
        .def("take_lease_served", &ENGINE_T::take_lease_served)
        .def("clear_read_lease_addr", &ENGINE_T::clear_read_lease_addr)
        .def("clear_read_lease", &ENGINE_T::clear_read_lease)
        .def("set_nmi", &ENGINE_T::set_nmi)
        .def("set_irq", &ENGINE_T::set_irq)
        .def("clear_irq", &ENGINE_T::clear_irq)
        .def("inject_write", &ENGINE_T::inject_write)
        .def("inject_region", &ENGINE_T::inject_region)
        .def("map_region", &ENGINE_T::map_region)
        .def("seed_image", &ENGINE_T::seed_image)
        .def("set_mapper_mmc3", &ENGINE_T::set_mapper_mmc3)
        .def("set_mapper_mmc1", &ENGINE_T::set_mapper_mmc1)
        .def("clear_mapper", &ENGINE_T::clear_mapper)
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


# Maximum instructions retired by one execution model; keep in sync with
# SEQ_MAX_K in SEQ_HOOK.
SEQ_MAX_K = 32


def emit_recurrence_dispatch(emit, exp, E: str, step_call: str, indent: str) -> None:
    recurrence = exp.get("recurrence")
    if recurrence is None:
        return
    cycles = recurrence.cycles_per_iteration
    prefix = recurrence.cycles_prefix
    if recurrence.kind == "identity":
        emit(f"{indent}long recurrence_n = max_steps - i;")
        emit(f"{indent}if (max_cycles > 0) {{")
        emit(f"{indent}    const long room = max_cycles - cycles_this;")
        emit(f"{indent}    const long cap = room > {prefix}"
             f" ? (room - {prefix} - 1) / {cycles} + 1 : 0;")
        emit(f"{indent}    if (recurrence_n > cap) recurrence_n = cap;")
        emit(f"{indent}}}")
        emit(f"{indent}{E}seq_cycles = recurrence_n * {cycles};")
        emit(f"{indent}return (int)recurrence_n;")
        return
    closure = (
        f"{E}ram[({E}st_pc + {recurrence.closure_operand_offset}) & 0xFFFF] "
        f"== 0x{recurrence.closure_operand_value:02X}"
    )
    if recurrence.kind == "invariant_poll":
        steps = recurrence.steps_per_iteration
        emit(f"{indent}if ({closure}) {{")
        emit(f"{indent}    const int recurrence_head = {E}st_pc;")
        emit(f"{indent}    {step_call}")
        emit(f"{indent}    long recurrence_n = 1;")
        emit(f"{indent}    const int recurrence_read = {E}shadow_read_target();")
        emit(f"{indent}    const int recurrence_lease = "
             f"{E}lease_slot(recurrence_read);")
        emit(f"{indent}    const bool recurrence_stable = "
             f"!{E}in_ranges({E}rwatch, recurrence_read)"
             f" || recurrence_lease < 0 || {E}lease_n[recurrence_lease] == 0;")
        emit(f"{indent}    if ({E}st_pc == recurrence_head && recurrence_stable) {{")
        emit(f"{indent}        recurrence_n = (max_steps - i) / {steps};")
        emit(f"{indent}        if (max_cycles > 0) {{")
        emit(f"{indent}            const long room = max_cycles - cycles_this;")
        emit(f"{indent}            const long cap = room > {prefix}"
             f" ? (room - {prefix} - 1) / {cycles} + 1 : 0;")
        emit(f"{indent}            if (recurrence_n > cap) recurrence_n = cap;")
        emit(f"{indent}        }}")
        emit(f"{indent}    }}")
        emit(f"{indent}    {E}seq_cycles = recurrence_n * {cycles};")
        emit(f"{indent}    return (int)(recurrence_n * {steps});")
        emit(f"{indent}}}")
        return
    if recurrence.kind != "affine_until":
        raise RuntimeError(f"unknown recurrence kind {recurrence.kind!r}")
    register = f"{E}st_{recurrence.register}"
    steps = recurrence.steps_per_iteration
    delta = recurrence.delta
    emit(f"{indent}if ({closure}) {{")
    emit(f"{indent}    const int recurrence_head = {E}st_pc;")
    emit(f"{indent}    const int recurrence_initial = {register};")
    if delta < 0:
        emit(f"{indent}    const long recurrence_terminal = "
             "recurrence_initial ? recurrence_initial : 256;")
    else:
        emit(f"{indent}    const long recurrence_terminal = "
             "recurrence_initial ? 256 - recurrence_initial : 256;")
    emit(f"{indent}    long recurrence_n = recurrence_terminal;")
    emit(f"{indent}    const long step_cap = (max_steps - i) / {steps};")
    emit(f"{indent}    if (recurrence_n > step_cap) recurrence_n = step_cap;")
    emit(f"{indent}    if (max_cycles > 0) {{")
    emit(f"{indent}        const long room = max_cycles - cycles_this;")
    emit(f"{indent}        const long cycle_cap = room > {prefix}"
         f" ? (room - {prefix} - 1) / {cycles} + 1 : 0;")
    emit(f"{indent}        if (recurrence_n > cycle_cap) recurrence_n = cycle_cap;")
    emit(f"{indent}    }}")
    emit(f"{indent}    {step_call}")
    emit(f"{indent}    if (recurrence_n > 1) {{")
    emit(f"{indent}        {register} = "
         f"(recurrence_initial + {delta} * recurrence_n) & 0xFF;")
    emit(f"{indent}        {E}st_p = ({E}st_p & ~0x82) | 0x20"
         f" | ({register} == 0 ? 0x02 : 0) | ({register} & 0x80);")
    emit(f"{indent}        {E}st_pc = recurrence_n == recurrence_terminal"
         f" ? (recurrence_head + {recurrence.fallthrough_offset}) & 0xFFFF"
         " : recurrence_head;")
    emit(f"{indent}    }}")
    emit(f"{indent}    {E}seq_cycles = recurrence_n * {cycles};")
    emit(f"{indent}    return (int)(recurrence_n * {steps});")
    emit(f"{indent}}}")


def assemble_seq_module(lines, body, experts, core_digest, elem,
                         n_seen, emit_step_core) -> tuple[str, int, bool]:
    """Assemble the per-ROM sequence-module TU from the emitted expert
    bodies: a SEQMOD_T struct compiled against the core's generated header,
    exporting its dispatch as a raw function pointer for attach_seq()."""
    emit = lines.append
    core_name = f"Engine_{core_digest}"
    mod_xbuf = max(e["model"].tok.weight.shape[1] for e in experts)
    max_k = max(e["n_steps"] for e in experts)
    if max_k > SEQ_MAX_K:
        raise RuntimeError(
            f"sequence window of {max_k} steps exceeds the core's "
            f"execution-model limit ({SEQ_MAX_K})"
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
    # matched, no deliverable interrupt, enough step/cycle budget.  A
    # switch on st_pc keeps dispatch O(log windows); several banks may
    # carry different windows at one head, so byte checks arbitrate
    # within a case.
    by_head: dict[int, list] = {}
    for exp in experts:
        by_head.setdefault(exp["head"], []).append(exp)
    emit("")
    emit("    int dispatch(Eng& E, long i, long max_steps, long cycles_this,")
    emit("                 long max_cycles, bool stop_on_trap) {")
    emit("        if (E.nmi_pending || E.irq_ready() || stop_on_trap) return 0;")
    emit("        switch (E.st_pc) {")
    for head in sorted(by_head):
        emit(f"        case 0x{head:04X}:")
        for exp in by_head[head]:
            window = exp["window"]
            k, prefix = exp["n_steps"], exp["cycles_prefix"]
            if exp.get("recurrence") is not None and exp["recurrence"].kind == "identity":
                k, prefix = 1, 0
            match = " && ".join(
                f"E.ram[0x{(head + j) & 0xFFFF:04X}] == 0x{b:02X}"
                for j, b in enumerate(window)
            )
            if exp.get("has_stores"):
                # Licensed store targets stay below $2000; refuse
                # dispatch while any write watch reaches below it.
                match = f"E.watch_write_lo >= 0x2000 && {match}"
            emit(f"            if (i + {k} <= max_steps")
            emit(f"                && (max_cycles <= 0 || cycles_this + {prefix} < max_cycles)")
            emit(f"                && {match}) {{")
            emit_recurrence_dispatch(
                emit, exp, "E.",
                f"step_core__{exp['name']}(E);",
                "                ",
            )
            emit(f"                step_core__{exp['name']}(E);")
            emit(f"                E.seq_cycles = {exp['cycles']};")
            emit(f"                return {k};")
            emit("            }")
        emit("            return 0;")
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


class EmitterContext:
    """Shared emission state: the union pass allocates (hulls, memory
    targets, registers), expert passes resolve onto it, the epilogue
    emits from it.  Also the dialect: element type and member prefix
    ("" for the engine, "E." for a sequence module)."""

    def __init__(self, *, elem: str, E: str, inv_scale: str):
        self.elem = elem
        self.E = E
        self.inv_scale = inv_scale
        self.hull_tiebreak: list[int] = []  # flat hull id -> latest?
        self.mem_hulls: list[list[int]] = [[] for _ in range(16)]
        self.mem_fallback_hulls: list[int] = []
        self.n_memory_heads = 0
        self.n_hull_heads = 0  # heads still on real hulls
        # Running-argmax registers for constant-query heads, keyed by
        # (union lookup id, value pair): the union pass allocates in
        # head-scan order, expert passes resolve through fold provenance.
        self.reg_of: dict[tuple[int, int], int] = {}

    def xref(self, i):
        return f"x[{i}]"

    def dref(self, i):
        return f"({self.inv_scale}*(double)x[{i}])"


def row_int(csr_m, r, ref, where):
    return expr_int(*row(csr_m, r), ref, where)


def row_raw(indices, data, ref, where):
    return expr_int(indices, data, ref, where)


def plan_model(ctx, mdl, label, union_lookup_of=None):
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
    is_union = union_lookup_of is None
    dm = mdl.tok.weight.shape[1]
    nh = mdl.attn[0].num_heads if mdl.attn else 0
    plans: list[dict] = []
    for li in range(len(mdl.attn)):
        check_ffn_kinds(mdl, li)
        ip = csr(mdl.attn[li].in_proj_weight)
        op = csr(mdl.attn[li].out_proj.weight)
        fi = csr(mdl.ff_in[li].weight)
        fo = csr(mdl.ff_out[li].weight)
        ffn = fo.shape[1]
        kinds = mdl.head_kinds[li]
        ffn_kinds = mdl.ffn_kinds[li]
        tiebreak = mdl.head_tiebreak[li]
        groups = mdl.head_cache_groups[li]
        mem_groups = mdl.head_memory_groups[li]
        lookup_ids = mdl.head_lookup_ids[li]
        pairs = mdl.head_pairs[li]
        memory_outputs = getattr(
            mdl, "head_memory_outputs", [[None] * nh for _ in mdl.attn]
        )[li]

        lookup = [h for h in range(nh) if kinds[h] == 1]
        passthrough = [h for h in range(nh) if kinds[h] == 2]
        for h in lookup:
            if not tiebreak[h]:
                raise RuntimeError(
                    f"{label} L{li} head {h}: AVERAGE tie-break can return "
                    "fractional hull outputs"
                )

        # Shared hulls, first-seen in ascending head order (engine order).
        # Expert hull ids exist only for the hull_seen live-insert guard;
        # only the union's memory hulls receive insert_write tokens.
        hull_base = len(ctx.hull_tiebreak)
        hull_of_group: dict[int, int] = {}
        hull_of_head: dict[int, int] = {}
        for h in lookup:
            if mem_groups[h] == MEMORY_GROUP_FUSED:
                continue
            g = groups[h]
            if g < 0:
                raise RuntimeError(f"{label} L{li} head {h}: negative cache group")
            if g not in hull_of_group:
                hull_of_group[g] = hull_base + len(hull_of_group)
                ctx.hull_tiebreak.append(int(tiebreak[h]))
            hull_of_head[h] = hull_of_group[g]
            if mem_groups[h] >= 0 and is_union:
                ctx.mem_hulls[mem_groups[h]].append(hull_of_head[h])
                ctx.n_memory_heads += 1

        # Structural passthrough/erase fusion: fold kind-2 heads and
        # neurons into direct D->D residual maps (no hull/ReGLU needed).
        pt_cols = [2 * h + c for h in passthrough for c in range(2)]
        attn_direct = compose(op, pt_cols, ip[[2 * dm + c for c in pt_cols], :])
        op_kept = zero_columns(op, pt_cols)

        lookup_cols = {2 * h + c for h in lookup for c in range(2)}
        for c in np.unique(op_kept.indices):
            if int(c) not in lookup_cols:
                raise RuntimeError(
                    f"{label} L{li}: out_proj column {int(c)} survives at a "
                    "non-lookup head slot"
                )

        pt_j = [j for j in range(ffn) if ffn_kinds[j] == 2]
        ffn_direct = compose(fo, pt_j, fi[[ffn + j for j in pt_j], :])
        fo_kept = zero_columns(fo, pt_j)
        live = sorted(int(j) for j in np.unique(fo_kept.indices))

        # Per-head emission blocks (each self-contained: it defines
        # ho{2h} and ho{2h+1}), keyed by head.
        head_blocks: dict[int, list[str]] = {}
        memory_read_blocks: list[str] = []
        memory_read_names: dict[int, tuple[str, str, str]] = {}

        # ── Attention: hull heads, then residual updates ─────────
        for h in lookup:
            b = head_blocks.setdefault(h, [])
            hid = hull_of_head.get(h)
            tb = "TieBreak::LATEST" if tiebreak[h] else "TieBreak::AVERAGE"
            # Eligibility decisions live in model/classify.py, shared
            # in model/classify.py, the single license source.
            q0_idx, q0_dat = row(ip, 2 * h)
            q1_idx, q1_dat = row(ip, 2 * h + 1)
            qx_terms = list(zip(q0_dat, q0_idx, strict=True))
            qy_terms = list(zip(q1_dat, q1_idx, strict=True))
            mem_terms = None
            fused = mem_groups[h] == MEMORY_GROUP_FUSED
            if mem_groups[h] >= 0 or fused:
                mem_terms = memory_query(
                    qx_terms, qy_terms, one_id=mdl.one_slot,
                )
            mem_ok = mem_terms is not None
            if not is_union and (mem_groups[h] >= 0 or fused) and not mem_ok:
                raise RuntimeError(
                    f"{label} L{li} head {h}: memory head is not "
                    "structure-eligible; a per-expert hull would diverge"
                )
            if fused and is_union:
                raise RuntimeError(
                    f"{label} L{li} head {h}: fused reads are expert-only"
                )

            reg_q = None
            reg_score = None
            v0_i = v1_i = None
            if not mem_ok:
                reg_q = register_query(qx_terms, qy_terms,
                                       one_id=mdl.one_slot,
                                       latest=bool(tiebreak[h]))
                if reg_q is not None:
                    # The integer score form is part of the register
                    # license: qx == qy > 0 makes the score a positive
                    # multiple of kx + ky, and the clear gate an exact
                    # zero test -- no f64 in the fold or the compare.
                    k0_idx, k0_dat = row(ip, dm + 2 * h)
                    k1_idx, k1_dat = row(ip, dm + 2 * h + 1)
                    reg_score = integer_score_terms(
                        reg_q[0], reg_q[1],
                        list(zip(k0_dat, k0_idx, strict=True)),
                        list(zip(k1_dat, k1_idx, strict=True)),
                    )
                    if reg_score is None:
                        reg_q = None
                if reg_q is not None:
                    try:
                        v0_i = expr_int(*row(ip, 2 * dm + 2 * h),
                                         ctx.xref, f"{label} L{li} reg v0")
                        v1_i = expr_int(*row(ip, 2 * dm + 2 * h + 1),
                                         ctx.xref, f"{label} L{li} reg v1")
                    except RuntimeError:
                        reg_q = None
            reg_ok = reg_q is not None

            if mem_ok:
                g = mem_groups[h]
                assert mem_terms is not None  # mem_ok on this branch
                qi_expr = expr_int([i for _c, i in mem_terms],
                                    [c for c, _i in mem_terms],
                                    ctx.xref, f"{label} L{li} mem query")
                b.append(f"        {ctx.elem} ho{2 * h}, ho{2 * h + 1};")
                b.append("        {")
                if fused:
                    read_id = lookup_ids[h]
                    names = memory_read_names.get(read_id)
                    if names is None:
                        stem = f"mr{len(memory_read_names)}"
                        names = (f"{stem}_q", f"{stem}_v", f"{stem}_a")
                        memory_read_names[read_id] = names
                        qname, vname, aname = names
                        memory_read_blocks.extend((
                            f"        const long long {qname} = shr2({qi_expr});",
                            f"        const bool {stem}_valid = "
                            f"(unsigned long long){qname} <= 0xFFFFULL;",
                            f"        const long long {vname} = {stem}_valid ? "
                            f"{ctx.E}ram[(int){qname}] : 0;",
                            f"        const long long {aname} = "
                            f"{stem}_valid ? {qname} : 0;",
                        ))
                    _qname, vname, aname = names
                else:
                    assert hid is not None
                    b.append(f"            const long long qi = shr2({qi_expr});")
                    k1_idx, k1_dat = row(ip, dm + 2 * h + 1)
                    guard = integer_score_terms(
                        1.0, 1.0, [],
                        list(zip(k1_dat, k1_idx, strict=True)),
                    )
                    b.append(f"            if (hull_seen[{hid}] != {ctx.E}seq) {{")
                    if guard is not None:
                        # A live (unrescinded) K row means the model would
                        # insert into a memory hull, which this engine cannot
                        # do; the clear gate is an exact zero test.
                        live_cond = " && ".join(
                            f"x[{g_id}] == 0" for g_id in guard[1]
                        ) or "true"
                        b.append(f"                if ({live_cond})")
                    else:
                        k1 = expr_dbl(*row(ip, dm + 2 * h + 1), ctx.dref)
                        b.append(f"                const double kyg = {k1};")
                        b.append("                if (kyg > -1.0e29)")
                    b.append('                    throw std::runtime_error('
                             '"live forward insert into a memory-group hull");')
                    b.append(f"                hull_seen[{hid}] = {ctx.E}seq;")
                    b.append("            }")
                    b.append("            long long mo[2] = {0, 0};")
                    # Group dispatch: qi is the address minus this group's
                    # base, so qi in [0, 4096) iff the address falls in this
                    # group.  Other groups' contributions are gate-zeroed
                    # downstream, and (0,0) is exactly the empty-group result.
                    b.append("            if ((unsigned long long)qi >> 12 == 0)")
                    b.append(f"                {ctx.E}mem_query({g}, qi, mo);")
                outputs = memory_outputs[h] or ("value", "address")

                def served_output(
                    kind,
                    value_name=vname if fused else "mo[0]",
                    address_name=aname if fused else "mo[1]",
                ):
                    if kind == "value":
                        return value_name
                    if kind == "address":
                        return address_name
                    if kind.startswith("bit"):
                        return f"(({value_name} >> {int(kind[3:])}) & 1)"
                    raise RuntimeError(f"unknown memory output {kind!r}")

                for component in range(2):
                    value = (
                        served_output(outputs[component])
                        if component < len(outputs) else "0"
                    )
                    b.append(
                        f"            ho{2 * h + component} = "
                        f"({ctx.elem})({value} << F);"
                    )
                b.append("        }")
            elif reg_ok:
                # Register index: allocated by the union in scan order,
                # resolved by experts through the fold's provenance so
                # every model folds into / reads the same register.
                lu_id, pair = lookup_ids[h], pairs[h]
                if is_union:
                    key = (lu_id, pair)
                    if key in ctx.reg_of:
                        raise RuntimeError(
                            f"union L{li} head {h}: register key {key} repeated"
                        )
                    r = ctx.reg_of[key] = len(ctx.reg_of)
                else:
                    assert union_lookup_of is not None  # expert pass
                    r = ctx.reg_of.get((union_lookup_of.get(lu_id, -1), pair))
                    if r is None:
                        raise RuntimeError(
                            f"{label} L{li} head {h}: constant-query head "
                            f"has no union register (lookup {lu_id} pair {pair})"
                        )
                assert reg_score is not None  # reg_ok on this branch
                score_terms, gate_ids = reg_score
                s_expr = expr_int(
                    [i for _c, i in score_terms],
                    [c for c, _i in score_terms],
                    ctx.xref, f"{label} L{li} reg score",
                )
                live_cond = " && ".join(
                    f"x[{g_id}] == 0" for g_id in gate_ids
                ) or "true"
                b.append(f"        {ctx.elem} ho{2 * h}, ho{2 * h + 1};")
                b.append("        {")
                b.append(f"            if ({live_cond}) {{  // live: fold into the running argmax")
                b.append(f"                const long long s = {s_expr};")
                b.append(f"                if (!{ctx.E}reg_any[{r}] || s >= {ctx.E}reg_s[{r}]) {{")
                b.append(f"                    {ctx.E}reg_any[{r}] = true;")
                b.append(f"                    {ctx.E}reg_s[{r}] = s;")
                b.append(f"                    {ctx.E}reg_v0[{r}] = {v0_i};")
                b.append(f"                    {ctx.E}reg_v1[{r}] = {v1_i};")
                b.append("                }")
                b.append("            }")
                b.append(f"            ho{2 * h} = {ctx.E}reg_any[{r}] ? {ctx.E}reg_v0[{r}] : 0;")
                b.append(f"            ho{2 * h + 1} = {ctx.E}reg_any[{r}] ? {ctx.E}reg_v1[{r}] : 0;")
                b.append("        }")
            else:
                if not is_union:
                    raise RuntimeError(
                        f"{label} L{li} head {h}: expert lookup head needs a "
                        "real hull; refusing (its history would diverge from "
                        "the union's across routed steps)"
                    )
                ctx.n_hull_heads += 1
                if mem_groups[h] >= 0:
                    ctx.mem_fallback_hulls.append(hid)
                # Q/K/V straight to double: the hull resolves in double.
                q0 = expr_dbl(*row(ip, 2 * h), ctx.dref)
                q1 = expr_dbl(*row(ip, 2 * h + 1), ctx.dref)
                k0 = expr_dbl(*row(ip, dm + 2 * h), ctx.dref)
                k1 = expr_dbl(*row(ip, dm + 2 * h + 1), ctx.dref)
                v0 = expr_dbl(*row(ip, 2 * dm + 2 * h), ctx.dref)
                v1 = expr_dbl(*row(ip, 2 * dm + 2 * h + 1), ctx.dref)
                b.append(f"        {ctx.elem} ho{2 * h}, ho{2 * h + 1};")
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
                b.append(f"            ho{2 * h} = ({ctx.elem})t0;")
                b.append(f"            ho{2 * h + 1} = ({ctx.elem})t1;")
                b.append(f"            if ((double)t0 != s0 || (double)t1 != s1 "
                         f"|| ho{2 * h} != t0 || ho{2 * h + 1} != t1)")
                b.append('                throw std::runtime_error('
                         '"hull output not representable at the fixed-point '
                         'scale");')
                b.append("        }")
            b.append(f"        (void)ho{2 * h}; (void)ho{2 * h + 1};")

        # attn_direct reads pre-attention x; out_proj rows read the
        # hull outputs ho{..} and apply first.
        ad_rows = [r for r in range(dm)
                   if attn_direct.indptr[r + 1] > attn_direct.indptr[r]]
        opkept: list[tuple[int, list[int], list[float]]] = []
        for r in range(dm):
            if op_kept.indptr[r + 1] > op_kept.indptr[r]:
                idxs, dat = row(op_kept, r)
                opkept.append((r, [int(c) for c in idxs], list(dat)))
        ad_def, ad_apply = fuse_direct(
            attn_direct, ad_rows, "ad", (r for r, _i, _d in opkept),
            f"L{li} attn_direct", elem=ctx.elem, xref=ctx.xref,
        )

        # ── FFN: direct map and activations read post-attention x ─
        fd_rows = [r for r in range(dm)
                   if ffn_direct.indptr[r + 1] > ffn_direct.indptr[r]]
        neuron_blocks: dict[int, list[str]] = {}
        for j in live:
            g = row_int(fi, j, ctx.xref, f"L{li} ff_gate")
            v = row_int(fi, ffn + j, ctx.xref, f"L{li} ff_value")
            nb = [f"        {ctx.elem} a{j};"]
            nb.append(f"        {{ const {ctx.elem} g = {g}; "
                      f"const {ctx.elem} v = {v}; "
                      f"a{j} = g > 0 ? shr(g*v) : 0; }}")
            neuron_blocks[j] = nb
        # ff_out rows read the ReGLU activations a{..}.
        fokept: list[tuple[int, list[int], list[float]]] = []
        for r in range(dm):
            if fo_kept.indptr[r + 1] > fo_kept.indptr[r]:
                idxs, dat = row(fo_kept, r)
                fokept.append((r, [int(c) for c in idxs], list(dat)))
        fd_def, fd_apply = fuse_direct(
            ffn_direct, fd_rows, "fd", (r for r, _i, _d in fokept),
            f"L{li} ffn_direct", elem=ctx.elem, xref=ctx.xref,
        )

        plans.append({
            "li": li, "lookup": lookup, "head_blocks": head_blocks,
            "memory_reads": memory_read_blocks,
            "ad_def": ad_def, "opkept": opkept, "ad_apply": ad_apply,
            "fd_def": fd_def, "live": live, "neuron_blocks": neuron_blocks,
            "fokept": fokept, "fd_apply": fd_apply,
        })
    return plans



def generate_source(model, step_spec: dict | None = None, experts=None,
                    seq_module_core: str | None = None) -> tuple[str, int, bool]:
    """Return ``(source, memory_head_count, has_machine_step)``.

    ``step_spec`` enables the native machine protocol. ``experts`` contains
    independently folded opcode or sequence models whose attention state is
    resolved through union lookup provenance. ``seq_module_core`` switches
    emission from the core engine to a per-ROM module attached to that core.
    """
    experts = experts or []
    seq_mode = seq_module_core is not None
    if not seq_mode and any(e.get("kind") == "seq" for e in experts):
        raise RuntimeError(
            "sequence experts are compiled per-ROM (SequenceModule) and "
            "attached at runtime, not emitted into the core engine"
        )
    if seq_mode and (
        not experts or any(e.get("kind") != "seq" for e in experts)
    ):
        raise RuntimeError("a sequence module takes kind='seq' experts only")
    if experts and step_spec is None:
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

    elem = "int"
    f_bits = 1
    inv_scale = f"0x1p-{f_bits}"
    ctx = EmitterContext(elem=elem, E=E, inv_scale=inv_scale)

    layer_plans = plan_model(ctx, model, "union")
    expert_plans = [
        (exp, plan_model(ctx, exp["model"], f"expert {exp['name']}",
                         exp["union_lookup_of"]))
        for exp in experts
    ]
    hull_tiebreak = ctx.hull_tiebreak
    mem_hulls = ctx.mem_hulls
    mem_fallback_hulls = ctx.mem_fallback_hulls
    n_memory_heads = ctx.n_memory_heads
    n_hull_heads = ctx.n_hull_heads
    n_regs = len(ctx.reg_of)
    layer_sig = f"{elem}* x"
    if seq_mode:
        layer_sig = f"Eng& E, {layer_sig}"

    # ── Emit the layer functions: one body per model -- the union stack
    # plus each rebuilt expert's own (smaller) stack.  Expert bodies are
    # complete emissions of their own weight matrices, not pruned copies of
    # the union's; the fold already removed the dead work at graph level. ──
    def emit_layer_fn(plan, suffix):
        li = plan["li"]
        out = [f"    void layer_{li}{suffix}({layer_sig}) {{"]
        out.extend(plan["memory_reads"])
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
    one_e, zero_l = "SCALE", "0"

    def sc(v, e=elem):
        return f"(({e})({v}) << F)"

    def rdv(slot):
        return f"shr2(x[{slot}])"

    def emit_step_core(fn_name, layer_suffix, mdl, spec):
        # One model forward, sharing the embed/readout/write/halt protocol;
        # layer_suffix selects the union stack ("") or a routed expert's
        # rebuilt stack ("__<name>").  Sequence models retire a licensed
        # region through one final-state receipt; ordinary experts retire one
        # instruction.  Persistent st_* registers are the model interface.
        ins = spec["inputs"]
        sequence_final = spec.get("sequence_final")
        step_outs = None if sequence_final else (
            spec.get("step_outputs") or [spec["outputs"]]
        )
        if not sequence_final:
            hc, hv = spec["halt_head"]
            shc, shv = spec["state_head"]
        dm = mdl.tok.weight.shape[1]

        args = "Eng& E" if seq_mode else ""
        if not sequence_final:
            args += (", " if args else "") + "long long* out"
        emit(f"    void {fn_name}({args}) {{")
        emit(f"        {elem}* x = xbuf;")
        emit(f"        for (int i = 0; i < {dm}; ++i) x[i] = 0;")
        emit(f"        x[{mdl.one_slot}] = {one_e};")

        def embed(key, value):
            if key in ins:
                emit(f"        x[{ins[key]}] = {value};")

        embed("is_state", one_e)
        embed("pc_lo", sc(f"{E}st_pc & 0xFF"))
        embed("pc_hi", sc(f"({E}st_pc >> 8) & 0xFF"))
        embed("a", sc(f"{E}st_a"))
        embed("x", sc(f"{E}st_x"))
        embed("y", sc(f"{E}st_y"))
        embed("s", sc(f"{E}st_s"))
        for slot, mask in spec["p_flags"]:
            emit(f"        x[{slot}] = ({E}st_p & {mask}) ? {one_e} : {zero_l};")
        embed("ir", sc(f"{E}st_ir"))
        embed("oplo", sc(f"{E}st_oplo"))
        embed("ophi", sc(f"{E}st_ophi"))
        embed("ea_lo", sc(f"{E}st_ea & 0xFF"))
        embed("ea_hi", sc(f"({E}st_ea >> 8) & 0xFF"))
        embed("data", sc(f"{E}st_data"))
        embed("halted", sc(f"{E}st_halted ? 1 : 0"))
        for bit in range(8):
            embed(f"a_bit{bit}", sc(f"({E}st_a >> {bit}) & 1"))
        if "nmi" in ins:
            embed("nmi", sc(f"{E}nmi_pending ? 1 : 0"))
            emit(f"        {E}nmi_pending = false;")
        if "irq" in ins:
            emit(f"        const bool take_irq = {E}irq_ready();")
            embed("irq", sc("take_irq ? 1 : 0"))
            emit(f"        if (take_irq) {E}irq_pending = false;")
        emit(f"        ++{E}seq;")
        for li in range(len(mdl.attn)):
            emit(f"        layer_{li}{layer_suffix}("
                 f"{'E, ' if seq_mode else ''}x);")
        if sequence_final:
            outs = sequence_final["model"]
            constants = sequence_final["constant"]
            fields = {
                "new_a": ("st_a", 0xFF),
                "new_x": ("st_x", 0xFF),
                "new_y": ("st_y", 0xFF),
                "new_s": ("st_s", 0xFF),
            }
            for name, (field, mask) in fields.items():
                if name in outs:
                    emit(f"        {E}{field} = (int)({rdv(outs[name])} "
                         f"& 0x{mask:X});")
                elif name in constants:
                    emit(f"        {E}{field} = "
                         f"{int(constants[name]) & mask};")
            flag_masks = {
                "p_c": 0x01, "p_z": 0x02, "p_i": 0x04, "p_d": 0x08,
                "p_b": 0x10, "p_v": 0x40, "p_n": 0x80,
            }
            for name, mask in flag_masks.items():
                if name in outs:
                    value = f"({rdv(outs[name])} & 1) ? 0x{mask:X} : 0"
                elif name in constants:
                    value = f"0x{mask:X}" if constants[name] else "0"
                else:
                    continue
                emit(f"        {E}st_p = ({E}st_p & ~0x{mask:X}) | ({value});")
            control = spec["sequence_control"]
            if control["relative_to_head"]:
                fallthrough = f"({E}st_pc + {control['fallthrough']})"
                target = f"({E}st_pc + {control['target']})"
            else:
                fallthrough = str(control["fallthrough"])
                target = str(control["target"])
            operand_offset = control["relative_operand_offset"]
            if operand_offset is not None:
                target = (
                    f"({E}st_pc + {control['fallthrough']} + "
                    f"(int)(std::int8_t){E}ram[({E}st_pc + {operand_offset}) & 0xFFFF])"
                )
            predicate = control["predicate"]
            if control["flag_mask"]:
                mask = control["flag_mask"]
                expected = f"0x{mask:X}" if control["flag_value"] else "0"
                condition = f"({E}st_p & 0x{mask:X}) == {expected}"
            elif predicate is not None:
                condition = f"({rdv(predicate)} & 1)"
            else:
                condition = None
            successor = (
                fallthrough if condition is None
                else f"({condition} ? {target} : {fallthrough})"
            )
            emit(f"        {E}st_pc = (int)({successor} & 0xFFFF);")
            for step_writes in spec["sequence_writes"]:
                for addr_slot, value_slot in step_writes:
                    emit(f"        {E}insert_write((int)({rdv(addr_slot)} & 0xFFFF), "
                         f"(int)({rdv(value_slot)} & 0xFF), {E}seq + 1);")
            emit("    }")
            return
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
            emit(f"            {E}insert_write((int)out[{off + 15} + 2 * i], "
                 f"(int)out[{off + 16} + 2 * i], {E}seq + 1);")
            emit("        }")
            if k + 1 < len(step_outs):
                # Non-final rows of a sequence are generation-verified
                # non-halting; only the final row competes STATE vs HALT.
                emit(f"        out[{off + 13}] = 0;")
        last = 21 * (len(step_outs) - 1)
        emit(f"        const long long halt_logit = "
             f"{expr_int(hc, hv, lambda i: f'x[{i}]', 'halt head')};")
        emit(f"        const long long state_logit = "
             f"{expr_int(shc, shv, lambda i: f'x[{i}]', 'state head')};")
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
        return assemble_seq_module(
            lines, body, experts, seq_module_core, elem,
            len(hull_tiebreak), emit_step_core,
        )

    n_hulls = len(hull_tiebreak)
    # The C++ run loop (KernelBus shape) needs the same memory-write path the
    # machine_step uses, so it is emitted under the same condition.
    has_run = step_spec is not None and n_memory_heads > 0

    emit("// Generated exact-int32 engine (model/engine_gen.py).")
    emit(INCLUDES)
    emit("")
    emit("struct ENGINE_T {")
    emit(f"    static constexpr int D = {d_model};")
    emit("    long seq = -1;")
    # hull_seen doubles as the per-seq live-insert guard for classified
    # heads, so it survives even when no head keeps a real hull.
    if n_hull_heads:
        emit(f"    HardAttentionHead hulls[{n_hulls}];")
    emit(f"    long hull_seen[{max(n_hulls, 1)}];")
    emit(f"    static constexpr int F = {f_bits};")
    emit(f"    static constexpr {elem} SCALE = 1 << F;")
    emit("    static constexpr double SCALE_D = (double)SCALE;")
    emit(f"    {elem} xbuf[{xbuf_size}];")
    emit("")
    emit(FIXED_POINT_HELPERS.replace("ELEM", elem))
    emit("")
    emit(MEM_QUERY)
    if n_regs:
        emit("")
        emit("    // Running argmax for constant-query heads (exact integer")
        emit("    // scores; the query scale drops out of the ordering).")
        emit(f"    long long reg_s[{n_regs}];")
        emit(f"    {elem} reg_v0[{n_regs}], reg_v1[{n_regs}];")
        emit(f"    bool reg_any[{n_regs}];")
    if has_run:
        # Persistent-machine state for the C++ run loop (shared with the C++
        # run protocol in run_protocol.py).
        emit("")
        emit(run_state_members_cpp())
    emit("")
    emit("    ENGINE_T() { reset(); }")
    emit("")
    emit("    void reset() {")
    emit(f"        for (int i = 0; i < {n_hulls}; ++i) {{")
    if n_hull_heads:
        emit("            hulls[i].clear();")
    emit("            hull_seen[i] = -1;")
    emit("        }")
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
        emit("        st_halted = false; nmi_pending = false; irq_pending = false;")
        emit("        resume_read_skip = false;")
        emit("        clear_read_lease();")
        emit("        step_index = 0; cycles_total = 0;")
        emit("        clear_watches();")
        emit("        clear_mapper();")
    emit("        seq = -1;")
    emit("    }")
    emit("")
    lines.extend(body)
    emit("")
    emit("    void forward_token(FArr x_arr) {")
    emit('        if ((int)x_arr.size() != D) throw std::runtime_error("x size != d_model");')
    emit("        float* x = x_arr.mutable_data();")
    emit("        ++seq;")
    emit(f"        {elem}* xi = xbuf;")
    emit("        for (int i = 0; i < D; ++i)")
    emit(f"            xi[i] = ({elem})(long long)std::nearbyint((double)x[i] * SCALE_D);")
    for li in range(n_layers):
        emit(f"        layer_{li}(xi);")
    emit("        for (int i = 0; i < D; ++i) x[i] = (float)((double)xi[i] / SCALE_D);")
    emit("    }")
    emit("")
    fb_set = set(mem_fallback_hulls)
    fb_offs = [0]
    fb_flat: list[int] = []
    for g in range(16):
        fb_flat.extend(i for i in mem_hulls[g] if i in fb_set)
        fb_offs.append(len(fb_flat))

    emit("    void insert_write(int addr, int value, long pos) {")
    emit("        ++seq;")
    emit("        if (seq != pos)")
    emit('            throw std::runtime_error("direct WRITE position does not match engine sequence");')
    if has_run:
        emit("        ram[addr & 0xFFFF] = (unsigned char)(value & 0xFF);  // shadow-decode mirror")
    emit("        const int group = (addr >> 12) & 0xF;")
    emit("        const int wl = addr & 0xFFF;")
    emit("        mem_val[group][wl] = value & 0xFF;")
    emit("        mem_seq[group][wl] = pos;")
    emit("        mem_bits[group][wl >> 6] |= 1ULL << (wl & 63);")
    hull_targets = fb_flat
    hull_offs = fb_offs
    if hull_targets:
        emit("        const int local_i = addr - group * 4096;")
        emit("        const float local = (float)local_i;")
        emit("        const float kx_f = 2.0f * local;")
        emit("        const float ky_f = -local * local;")
        emit("        const double key[2] = {(double)kx_f, (double)ky_f};")
        emit("        const double val[2] = {(double)(value & 0xFF), (double)(addr & 0xFFFF)};")
        emit(f"        static const int offs[17] = {{{', '.join(map(str, hull_offs))}}};")
        emit(f"        static const int ids[{len(hull_targets)}] = {{{', '.join(str(i) for i in hull_targets)}}};")
        emit("        for (int i = offs[group]; i < offs[group + 1]; ++i) {")
        emit("            const int hid = ids[i];")
        emit("            if (hull_seen[hid] == seq) continue;")
        emit("            hulls[hid].insert(key, val, seq);")
        emit("            hull_seen[hid] = seq;")
        emit("        }")
    emit("    }")

    has_step = has_run
    if has_step:
        emit("")
        emit("    // One whole 6502 instruction from the persistent st_* state:")
        emit("    // STATE embed -> layers -> readout -> direct WRITE inserts ->")
        emit("    // halt logits.  Fills the 21-slot out buffer (trace row + next")
        emit("    // state + writes) and advances st_* to the next instruction.")
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
            emit("        if (!nmi_pending && !irq_ready()) {")
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
            emit("        if (!sid || nmi_pending || irq_ready() || stop_on_trap) return 0;")
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
                if exp.get("has_stores"):
                    # Licensed store targets stay below $2000; refuse
                    # dispatch while any write watch reaches below it.
                    checks = f"watch_write_lo >= 0x2000 && {checks}"
                emit(f"        case {j + 1}:  // {exp['name']}")
                guard_steps, guard_prefix = exp["n_steps"], exp["cycles_prefix"]
                if exp.get("recurrence") is not None and exp["recurrence"].kind == "identity":
                    guard_steps, guard_prefix = 1, 0
                emit(f"            if (i + {guard_steps} <= max_steps")
                emit(f"                && (max_cycles <= 0 || cycles_this + "
                     f"{guard_prefix} < max_cycles)")
                emit(f"                && {checks}) {{")
                emit("                ++fallback_count[st_pc];")
                emit_recurrence_dispatch(
                    emit, exp, "",
                    f"step_core__{exp['name']}();",
                    "                ",
                )
                emit(f"                step_core__{exp['name']}();")
                emit(f"                seq_cycles = {exp['cycles']};")
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
        emit(SEQ_HOOK)
        emit("")
        emit("    // Per-instruction entry point used by machine_step().")
        emit("    void machine_step(int pc, int a, int xr, int yr, int sr,")
        emit("                      int p, int ir, int oplo, int ophi,")
        emit("                      int ea, int data, int halted, int nmi,")
        emit("                      int irq, LArr out_arr) {")
        emit('        if ((int)out_arr.size() != 21) throw std::runtime_error("step out buffer must have 21 slots");')
        emit("        st_pc = pc & 0xFFFF; st_a = a & 0xFF; st_x = xr & 0xFF;")
        emit("        st_y = yr & 0xFF; st_s = sr & 0xFF; st_p = (p | 0x20) & 0xFF;")
        emit("        st_ir = ir & 0xFF; st_oplo = oplo & 0xFF; st_ophi = ophi & 0xFF;")
        emit("        st_ea = ea & 0xFFFF; st_data = data & 0xFF;")
        emit("        st_halted = halted != 0; nmi_pending = nmi != 0;")
        emit("        irq_pending = irq != 0;")
        emit("        step_core(out_arr.mutable_data());")
        emit("    }")

        # ── C++ run loop (KernelBus shape) and machine API ───────────
        emit("")
        emit(RUN_LOOP)

    emit("};")
    emit("")
    emit("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    emit('    py::class_<ENGINE_T>(m, "Engine", py::module_local())')
    emit("        .def(py::init<>())")
    emit("        .def(\"reset\", &ENGINE_T::reset)")
    emit("        .def(\"forward_token\", &ENGINE_T::forward_token)")
    if has_step:
        emit(PYBIND_STEP_METHODS)
    emit("        .def(\"insert_write\", &ENGINE_T::insert_write);")
    emit("}")

    source = "\n".join(lines) + "\n"
    return source, n_memory_heads, has_step


class GeneratedWeightsEngine:
    """The 6502 transformer as generated exact-int32 C++."""

    # Native methods bound straight off the pybind module (bind()): the hot
    # loops call them with no wrapper frame in between.
    machine_step: Callable
    run: Callable
    set_run_state: Callable
    get_run_state: Callable
    watch_read: Callable
    watch_write: Callable
    clear_watches: Callable
    set_write_journal: Callable
    set_read_lease: Callable
    set_read_lease_seq: Callable
    take_lease_served: Callable
    clear_read_lease: Callable
    clear_read_lease_addr: Callable
    set_nmi: Callable
    set_irq: Callable
    clear_irq: Callable
    engine_inject_write: Callable
    seed_image: Callable
    map_region: Callable
    set_mapper_mmc3: Callable
    set_mapper_mmc1: Callable
    clear_mapper: Callable
    read_ram: Callable
    dump_ram: Callable
    attach_seq: Callable
    detach_seq: Callable
    add_window: Callable
    clear_windows: Callable
    jit_hot: Callable
    window_shape: Callable
    digest: str

    def __init__(self, model, *, step_spec: dict | None = None,
                 experts=None, cache_key: str | None = None,
                 shapes_meta=None):
        self.n_experts = len(experts or [])
        # Shape-window scan metadata (id/signature/length/close rule) so a
        # cache-hit load can populate the runtime table without refolding.
        self.shapes_meta = list(shapes_meta or [])
        source, self.n_memory_heads, self.has_machine_step = generate_source(
            model, step_spec, experts=experts
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.digest = digest
        source = source.replace("ENGINE_T", f"Engine_{digest}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        n_shards = compile_shards()
        hpp_path = os.path.join(CACHE_DIR, f"engine_{digest}.hpp")
        shard_paths = [
            os.path.join(CACHE_DIR, f"engine_{digest}_s{k}.cpp")
            for k in range(n_shards)
        ]
        existing_shards = [
            os.path.join(CACHE_DIR, path)
            for path in os.listdir(CACHE_DIR)
            if path.startswith(f"engine_{digest}_s") and path.endswith(".cpp")
        ]
        if not os.path.exists(hpp_path) or set(existing_shards) != set(shard_paths):
            header, shards = shard_source(source, f"Engine_{digest}", n_shards)
            for path, body in zip(shard_paths, shards, strict=True):
                with open(path, "w") as f:
                    f.write(f'#include "engine_{digest}.hpp"\n\n{body}')
            with open(hpp_path, "w") as f:
                f.write(header)
            for path in set(existing_shards) - set(shard_paths):
                os.unlink(path)
        module, self.compile_seconds = load_engine_module(digest)
        if cache_key is not None:
            manifest_put(cache_key, {
                "digest": digest, "n_experts": self.n_experts,
                "n_memory_heads": self.n_memory_heads,
                "has_machine_step": self.has_machine_step,
                "shapes": self.shapes_meta,
                "compile_shards": n_shards,
            })
        self.bind(module)

    @classmethod
    def from_cache(cls, cache_key: str) -> GeneratedWeightsEngine | None:
        """Load an engine by config key alone, skipping the expert folds and
        the source emission entirely; None on any miss."""
        entry = manifest_read().get(cache_key)
        if entry is None:
            return None
        if entry.get("compile_shards") != compile_shards():
            return None
        if not os.path.exists(
            os.path.join(CACHE_DIR, f"engine_{entry['digest']}.hpp")
        ):
            return None
        self = cls.__new__(cls)
        self.digest = entry["digest"]
        self.n_experts = entry["n_experts"]
        self.n_memory_heads = entry["n_memory_heads"]
        self.has_machine_step = entry["has_machine_step"]
        self.shapes_meta = entry.get("shapes") or []
        module, self.compile_seconds = load_engine_module(entry["digest"])
        self.bind(module)
        return self

    def bind(self, module) -> None:
        self.engine = module.Engine()
        # The run loop is emitted under the same condition as machine_step.
        self.has_run_loop = self.has_machine_step
        if self.has_machine_step:
            # Bound methods exposed directly: the hot loops call them with no
            # wrapper frame in between.
            self.machine_step = self.engine.machine_step
            self.run = self.engine.run
            self.set_run_state = self.engine.set_run_state
            self.get_run_state = self.engine.get_run_state
            self.watch_read = self.engine.watch_read
            self.watch_write = self.engine.watch_write
            self.clear_watches = self.engine.clear_watches
            self.set_write_journal = self.engine.set_write_journal
            self.set_read_lease = self.engine.set_read_lease
            self.set_read_lease_seq = self.engine.set_read_lease_seq
            self.take_lease_served = self.engine.take_lease_served
            self.clear_read_lease = self.engine.clear_read_lease
            self.clear_read_lease_addr = self.engine.clear_read_lease_addr
            self.set_nmi = self.engine.set_nmi
            self.set_irq = self.engine.set_irq
            self.clear_irq = self.engine.clear_irq
            self.engine_inject_write = self.engine.inject_write
            self.seed_image = self.engine.seed_image
            self.map_region = self.engine.map_region
            self.set_mapper_mmc3 = self.engine.set_mapper_mmc3
            self.set_mapper_mmc1 = self.engine.set_mapper_mmc1
            self.clear_mapper = self.engine.clear_mapper
            self.read_ram = self.engine.read_ram
            self.dump_ram = self.engine.dump_ram
            self.attach_seq = self.engine.attach_seq
            self.detach_seq = self.engine.detach_seq
            self.add_window = self.engine.add_window
            self.clear_windows = self.engine.clear_windows
            self.jit_hot = self.engine.jit_hot
            self.window_shape = self.engine.window_shape

    def reset(self):
        self.engine.reset()

    def forward(self, x: np.ndarray) -> None:
        self.engine.forward_token(x)

    @property
    def supports_direct_writes(self) -> bool:
        return self.n_memory_heads > 0

    def insert_write(self, addr: int, value: int, pos: int) -> None:
        if self.n_memory_heads == 0:
            raise RuntimeError(
                "model has no memory-tagged lookup heads; a direct WRITE would "
                "be silently lost -- use a full WRITE-token forward instead"
            )
        self.engine.insert_write(addr, value, pos)

    def inject_region(self, base: int, data: bytes) -> None:
        if self.n_memory_heads == 0:
            raise RuntimeError(
                "model has no memory-tagged lookup heads; a direct WRITE would "
                "be silently lost -- use a full WRITE-token forward instead"
            )
        self.engine.inject_region(base, data)


class SequenceModule:
    """Per-ROM sequence-window module: the folded window experts compiled
    against a core engine's generated header (a single small TU, seconds to
    build, cached by digest), exporting its dispatch as a raw function
    pointer for the core's attach_seq() hook.  The holder must keep this
    object alive while it is attached."""

    def __init__(self, model, step_spec, experts, *, core_digest: str,
                 cache_key: str | None = None):
        source, _, _ = generate_source(
            model, step_spec, experts=experts,
            seq_module_core=core_digest,
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        source = source.replace("SEQMOD_T", f"SeqMod_{digest}")
        self.core_digest = core_digest
        self.digest = digest
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"seqmod_{digest}.cpp")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(source)
        module, self.compile_seconds = load_seqmod_module(digest)
        if cache_key is not None:
            manifest_put(cache_key, {
                "kind": "seqmod", "digest": digest,
                "core": core_digest, "n_windows": len(experts),
            })
        self.bind(module)

    @classmethod
    def from_cache(cls, cache_key: str, *, core_digest: str) -> SequenceModule | None:
        """Load a module by config key alone, skipping the sequence folds
        and the emission entirely; None on any miss."""
        entry = manifest_read().get(cache_key)
        if (
            entry is None
            or entry.get("kind") != "seqmod"
            or entry["core"] != core_digest
            or not os.path.exists(
                os.path.join(CACHE_DIR, f"seqmod_{entry['digest']}.cpp")
            )
        ):
            return None
        self = cls.__new__(cls)
        self.core_digest = core_digest
        self.digest = entry["digest"]
        module, self.compile_seconds = load_seqmod_module(entry["digest"])
        self.bind(module)
        return self

    def bind(self, module) -> None:
        if module.core_name != f"Engine_{self.core_digest}":
            raise RuntimeError(
                f"sequence module was compiled against {module.core_name}, "
                f"not Engine_{self.core_digest}"
            )
        self.max_k = int(module.max_k)
        self.mod = module.SeqModule()
        self.fn_ptr = int(module.SeqModule.fn_ptr())
        self.ctx_ptr = int(self.mod.ctx_ptr())
