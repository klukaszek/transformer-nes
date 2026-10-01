"""Emit model-specific C++ for analytic transformer weights.

Sparse indices and coefficients become C++ expressions, removing CSR gather
overhead. The residual uses exact fixed-point int32. Attention is served by
the exact memory structures and running-argmax registers licensed by
:mod:`model.classify`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from transformer_nes.cpu6502.backends.weights.run_protocol import (
    IN_RANGES_CPP,
    MACHINE_API_CPP,
    PYBIND_MACHINE_ABI,
    SHADOW_READ_TARGET_CPP,
    TRACE_RECEIPT_CPP,
    mem_query_cpp,
    run_loop_cpp,
    run_state_members_cpp,
)
from transformer_nes.files import write_atomic
from transformer_nes.graph.core import MEMORY_GROUP_FUSED
from transformer_nes.model.classify import (
    integer_score_terms,
    memory_query,
    register_query,
)
from transformer_nes.model.sources import source_digest

logger = logging.getLogger(__name__)

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

# What shapes the emitted engines (core and window modules): the runner
# that builds them and everything it imports (model/sources.py).
ENGINE_SOURCES = ("transformer_nes.cpu6502.backends.weights.runner",)


def engine_code_digest() -> str:
    return source_digest(*ENGINE_SOURCES)


def engine_config_key(*, bcd: bool, union_only: bool = False,
                      mirror_cpu_ram: bool = False, forward_hook: bool = False) -> str:
    """Cache key for a built engine, computable without folding or emitting:
    decimal mode plus every source file that shapes the fixed production core.
    A union-only engine holds no experts, so it keys separately."""
    raw = (f"{engine_code_digest()}|bcd={int(bcd)}|union_only={int(union_only)}"
           f"|mirror_cpu_ram={int(mirror_cpu_ram)}|forward_hook={int(forward_hook)}")
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def manifest_read() -> dict:
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def manifest_put(key: str, entry: dict) -> None:
    """Record a built artifact and prune every engine and window module
    built from older engine code: their keys can never be looked up again,
    so without this the cache grows by a full engine per edit.  (A PPU
    graph build is keyed by its own source and replaced under its key.)"""
    manifest = manifest_read()
    code = engine_code_digest()
    manifest[key] = {**entry, "code": None if entry.get("kind") == "graph" else code}
    manifest = {k: e for k, e in manifest.items() if e.get("code") in (None, code)}
    prune_cache({e["digest"] for e in manifest.values()})
    os.makedirs(CACHE_DIR, exist_ok=True)
    write_atomic(MANIFEST_PATH, json.dumps(manifest, indent=1, sort_keys=True).encode())


ARTIFACT_RE = re.compile(r"^(?:engine|seqmod|graph|build)_([0-9a-f]{16})(?:[_.].*)?$")


def prune_cache(live_digests: set[str]) -> None:
    """Delete engine/seqmod sources and build dirs of digests not in use."""
    if not os.path.isdir(CACHE_DIR):
        return
    for name in os.listdir(CACHE_DIR):
        m = ARTIFACT_RE.match(name)
        if m is None or m.group(1) in live_digests:
            continue
        path = os.path.join(CACHE_DIR, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.unlink(path)


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


def load_module(name: str, digest: str, sources: list[str]):
    """Load a cached extension or compile it from ``sources``."""
    if name in sys.modules:
        return sys.modules[name]
    build_dir = os.path.join(CACHE_DIR, f"build_{digest}")
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.exists(so_path):
        try:
            spec = importlib.util.spec_from_file_location(name, so_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
            return module
        except ImportError as exc:
            # A cached extension can outlive the process that provided its
            # dynamic-library search path (notably torch's libc10 on macOS).
            logger.info("rebuilding unavailable cached engine %s: %s", name, exc)
            os.unlink(so_path)

    import torch
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
            extra_ldflags=[
                f"-Wl,-rpath,{os.path.join(os.path.dirname(torch.__file__), 'lib')}"
            ],
            extra_include_paths=[CACHE_DIR],
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        if old_max_jobs is None:
            del os.environ["MAX_JOBS"]
    logger.info("compiled %s in %.1fs", name, time.perf_counter() - build_start)
    return module


def load_engine_module(digest: str):
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
    # Store forwarding and branch predictions fold differently: they key apart.
    tag = ",".join(
        f"{h:04X}:{bytes(c).hex()}x{r}{'f' if forward else ''}"
        + "".join("TF"[not t] for t in taken)
        for h, c, r, forward, taken in windows
    )
    raw = f"{engine_code_digest()}|core={core_digest}|bcd={int(bcd)}|seq={tag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_seqmod_module(digest: str):
    return load_module(
        f"seqmod_gen_{digest}",
        digest,
        [os.path.join(CACHE_DIR, f"seqmod_{digest}.cpp")],
    )


def join(terms) -> str:
    if not terms:
        return ""
    sign, first = terms[0]
    out = f"-{first}" if sign == "-" else first
    for sign, term in terms[1:]:
        out += f" {sign} {term}"
    return out


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
# Static C++ templates.

INCLUDES = """\
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <climits>
#include <memory>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

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

# Per-ROM sequence-module hook; pinned windows outrank canonical shapes.
SEQ_HOOK = """\
    typedef int (*SeqFn)(void*, void*, long, long, long, long, int);
    SeqFn seq_fn = nullptr;
    void* seq_ctx = nullptr;
    static constexpr int SEQ_MAX_K = 32;
    // The cycles the last expert forward retired: its model's output.
    long seq_cycles = 0;
    // The forward hook: when set, each expert's layers run outside, on
    // its embedded residual stream (ids: SEQ_HOOK_IDS in engine_gen.py);
    // it returns false for an expert it cannot run, which then runs here.
    bool (*forward_hook)(void*, int, int*) = nullptr;
    void* forward_hook_ctx = nullptr;
    void set_forward_hook(std::uintptr_t fn, std::uintptr_t ctx) {
        forward_hook = (bool (*)(void*, int, int*))fn;
        forward_hook_ctx = (void*)ctx;
    }
    std::uintptr_t ram_address() { return (std::uintptr_t)ram; }
    // Where the X-ray's read counting lives, for a hook that runs the
    // layers (and so the reads) itself: (&xray_on, xray_reads).
    std::pair<std::uintptr_t, std::uintptr_t> xray_address() {
        return {(std::uintptr_t)&xray_on, (std::uintptr_t)xray_reads};
    }
    // Module window heads: dispatch misses cost one array load instead
    // of a cross-module call into the module's head switch.
    unsigned char seq_head[65536] = {};

    // Instructions retired per execution path since the last xray_take():
    // sequence-module windows, shape experts, interrupt experts, opcode
    // experts, the union model; the dispatches (forwards, or a whole
    // spin or loop) that retired them; and per opcode, those that ran
    // singly.
    long xray_paths[5] = {};
    long xray_dispatches[5] = {};
    long xray_opcodes[256] = {};

    int seq_dispatch(long i, long max_steps, long cycles_this,
                     long max_cycles, bool stop_on_trap) {
        if (seq_fn && seq_head[st_pc]) {
            const int k = seq_fn(seq_ctx, this, i, max_steps, cycles_this,
                                 max_cycles, stop_on_trap ? 1 : 0);
            if (k) {
                xray_paths[0] += k;
                ++xray_dispatches[0];
                return k;
            }
        }
        const int k = shape_dispatch(i, max_steps, cycles_this, max_cycles,
                                     stop_on_trap);
        xray_paths[1] += k;
        xray_dispatches[1] += k > 0;
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

# Helpers of the run loop's step body: committing an expert's staged
# writes, and the per-step trace row (trace mode only).
STEP_HELPERS_CPP = """\
    // An interrupt or execution expert's staged writes, in order: into
    // memory, the trace, and the watched-write hits.  Returns the hits.
    int commit_writes(bool trace) {
        int hits = 0;
        for (int w = 0; w < exec_n_writes; ++w) {
            const int wa = exec_write_addr[w], wv = exec_write_value[w];
            if (trace) {
                w_row.push_back(step_index); w_addr.push_back(wa); w_value.push_back(wv);
            }
            insert_write(wa, wv, seq + 1);
            if (!wwatch.empty() && in_ranges(wwatch, wa)) {
                hit_addr.push_back(wa); hit_value.push_back(wv);
                ++hits;
            }
        }
        return hits;
    }
    void trace_row(int pc, int a, int x, int y, int s, int p, int ir, int oplo, int ophi,
                   int ea, int data, bool halted) {
        t_pc.push_back(pc); t_a.push_back(a); t_x.push_back(x); t_y.push_back(y);
        t_s.push_back(s); t_p.push_back(p); t_ir.push_back(ir); t_oplo.push_back(oplo);
        t_ophi.push_back(ophi); t_ea.push_back(ea); t_data.push_back(data);
        t_halted.push_back(halted ? 1 : 0);
    }"""

# Per-iteration body of the shared run loop (run_protocol.run_loop_cpp),
# in falling order of speed: a routed sequence window, an interrupt or
# per-opcode expert, else the full union model.
ENGINE_STEP_BODY = """\
            // A routed sequence consumes a whole window in one forward;
            // its guards (in seq_dispatch) keep this step-for-step
            // equivalent to the per-instruction path below.
            if (!trace) {
                const int k_seq = seq_dispatch(i, max_steps, cycles_this,
                                               max_cycles, stop_on_trap);
                if (k_seq > 0) {
                    fallthrough_pc = -1;
                    step_index += k_seq;
                    cycles_this += seq_cycles;
                    cycles_total += seq_cycles;
                    if (st_halted) { reason = "halt"; break; }
                    i += k_seq - 1;
                    if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }
                    continue;
                }
            }
            const bool was_int = nmi_pending || irq_ready();
            if (!was_int && st_pc != fallthrough_pc) ++block_entries[st_pc];
            const int prev_pc = st_pc;
            const unsigned char exec_opcode = ram[prev_pc];
            int pre_a = st_a, pre_x = st_x, pre_y = st_y, pre_p = st_p;
            int tr_lo = 0, tr_hi = 0, tr_ea = 0, tr_operand = 0;
            if (trace) {
                const int tr_len = TRACE_LENGTH[exec_opcode];
                if (tr_len > 1 || was_int) tr_lo = ram[(prev_pc + 1) & 0xFFFF];
                if (tr_len > 2 || was_int) tr_hi = ram[(prev_pc + 2) & 0xFFFF];
                // Interrupts report the interrupted PC with no effective address.
                if (!was_int) {
                    tr_ea = trace_ea(exec_opcode, prev_pc, pre_x, pre_y);
                    if (TRACE_DATA[exec_opcode] >= 11)
                        tr_operand = TRACE_MODE[exec_opcode] == 11 ? tr_lo : ram[tr_ea];
                }
            }
            if (was_int ? interrupt_step(nmi_pending) : execution_step(exec_opcode)) {
                ++xray_paths[was_int ? 2 : 3];
                ++xray_dispatches[was_int ? 2 : 3];
                if (!was_int) ++xray_opcodes[exec_opcode];
                const int tr_data = trace && !was_int
                    ? trace_data(exec_opcode, pre_a, pre_x, pre_y, pre_p, tr_operand)
                    : 0;
                const int hits = commit_writes(trace);
                if (trace) {
                    trace_row(prev_pc, st_a, st_x, st_y, st_s, st_p, exec_opcode, tr_lo, tr_hi,
                              tr_ea, tr_data, st_halted);
                    st_ir = exec_opcode; st_oplo = tr_lo; st_ophi = tr_hi;
                    st_ea = tr_ea; st_data = tr_data;
                }
                if (was_int) nmi_pending = false;
                fallthrough_pc = was_int || ENDS_WINDOW[exec_opcode]
                    ? -1 : (prev_pc + TRACE_LENGTH[exec_opcode]) & 0xFFFF;
                if (!was_int && ENDS_WINDOW[exec_opcode] == 2) {
                    ++branch_runs[prev_pc];
                    branch_taken[prev_pc] += st_pc != ((prev_pc + 2) & 0xFFFF);
                }
                ++step_index;
                const long step_cycles = seq_cycles;
                cycles_this += step_cycles; cycles_total += step_cycles;
                if (!was_int && stop_on_trap && st_pc == prev_pc) { reason = "trap"; break; }
                if (hits > 0) {
                    if (!was_int) st_ir = exec_opcode;  // the bus routes RMW writes by it
                    reason = "watch_write"; break;
                }
                if (st_halted) { reason = "halt"; break; }
                if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }
                continue;
            }
            // The full union model (undefined bytes, unfolded interrupts):
            // its receipt o[] carries the state, writes and stop flags.
            step_core(o);
            ++xray_paths[4];
            ++xray_dispatches[4];
            fallthrough_pc = was_int || ENDS_WINDOW[exec_opcode]
                ? -1 : (prev_pc + TRACE_LENGTH[exec_opcode]) & 0xFFFF;
            ++xray_opcodes[exec_opcode];
            const bool halted = o[11] != 0;
            if (trace)
                trace_row((int)o[0], (int)o[1], (int)o[2], (int)o[3], (int)o[4], (int)o[5],
                          (int)o[6], (int)o[7], (int)o[8], (int)o[9], (int)o[10], halted);
            int hits = 0;
            for (int w = 0; w < (int)o[14]; ++w) {
                const int wa = (int)o[15 + 2 * w], wv = (int)o[16 + 2 * w];
                if (trace) {
                    w_row.push_back(step_index); w_addr.push_back(wa); w_value.push_back(wv);
                }
                if (!wwatch.empty() && in_ranges(wwatch, wa)) {
                    hit_addr.push_back(wa); hit_value.push_back(wv);
                    ++hits;
                }
            }
            ++step_index;
            const long step_cycles = o[21];
            cycles_this += step_cycles; cycles_total += step_cycles;
            if (halted || o[13]) { st_halted = true; reason = "halt"; break; }
            if (stop_on_trap && (int)o[12] == prev_pc) { reason = "trap"; break; }
            if (hits > 0) { reason = "watch_write"; break; }
            if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }"""

# Shadow-decode + the shared run loop + machine API (run_protocol.py blocks;
# the run-loop protocol blocks).
RUN_LOOP = "\n\n".join((
    SHADOW_READ_TARGET_CPP,
    TRACE_RECEIPT_CPP,
    IN_RANGES_CPP,
    STEP_HELPERS_CPP,
    run_loop_cpp(extra_locals="        long long o[22];",
                 step_body=ENGINE_STEP_BODY),
    """\
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

    void map_bytes(int base, const unsigned char* d, long n) {
        for (long i = 0; i < n; ++i)
            map_write(base + (int)i, d[i]);
    }
    void map_region(int base, py::bytes data) {
        const std::string d = data;
        map_bytes(base, (const unsigned char*)d.data(), (long)d.size());
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
            if (MIRROR_CPU_RAM && a >= 0x0800 && a < 0x2000) continue;
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
        .def("set_nmi", &ENGINE_T::set_nmi)
        .def("set_irq", &ENGINE_T::set_irq)
        .def("clear_irq", &ENGINE_T::clear_irq)
        .def("inject_write", &ENGINE_T::inject_write)
        .def("inject_region", &ENGINE_T::inject_region)
        .def("map_region", &ENGINE_T::map_region)
        .def("seed_image", &ENGINE_T::seed_image)
        .def("read_ram", &ENGINE_T::read_ram)
        .def("dump_ram", &ENGINE_T::dump_ram)
        .def("set_run_state", &ENGINE_T::set_run_state)
        .def("get_run_state", &ENGINE_T::get_run_state)
        .def("attach_seq", &ENGINE_T::attach_seq)
        .def("set_forward_hook", &ENGINE_T::set_forward_hook)
        .def("ram_address", &ENGINE_T::ram_address)
        .def("xray_address", &ENGINE_T::xray_address)
        .def("detach_seq", &ENGINE_T::detach_seq)
        .def("add_window", &ENGINE_T::add_window)
        .def("clear_windows", &ENGINE_T::clear_windows)
""" + PYBIND_MACHINE_ABI


# Forward-hook ids: the core's experts are 0.., the attached sequence
# module's windows SEQ_HOOK_IDS.. (both in their build order).
SEQ_HOOK_IDS = 1000

# Maximum instructions retired by one execution model; keep in sync with
# SEQ_MAX_K in SEQ_HOOK.
SEQ_MAX_K = 32


def emit_window_dispatch(emit, exp, E: str, match: str, step_call: str) -> None:
    """Emit one window's dispatch case: the step and cycle budget guards,
    the byte match, then the forward, whose step_core leaves the window's
    cycles in seq_cycles.  The cycle guard uses a bound on the cycles
    before the last step, so a dispatched window holds only instructions
    single steps would have run.  A `JMP *` spin changes nothing per iteration: it retires
    as many as the budgets allow without a forward, overshooting the cycle
    budget by at most the last, as single steps would.  A loop window
    (experts.py:loop_recurrence) runs one iteration as a forward; if that
    looped back to the head, the following iterations cost the same
    model-computed cycles, and it retires as many as the budgets allow
    (a count loop all but its falling-through last)."""
    prefix = exp["cycles_prefix"]
    spin = exp["spin"]
    guards = exp.get("step_spec", {}).get("sequence_guards", ())
    for source, mask, expected in guards:
        if source == "state":
            match += f" && ({E}st_p & 0x{mask:02X}) == {mask if expected else 0}"
    steps = 1 if spin else exp["n_steps"]
    pad = "                "
    emit(f"            if (i + {steps} <= max_steps")
    emit(f"                && (max_cycles <= 0 || cycles_this + {0 if spin else prefix} < max_cycles)")
    emit(f"                && {match}) {{")
    loop = exp.get("loop")
    if spin:
        emit(f"{pad}long n = max_steps - i;")
        emit(f"{pad}if (max_cycles > 0 && n > (max_cycles - cycles_this - 1) / {spin} + 1)")
        emit(f"{pad}    n = (max_cycles - cycles_this - 1) / {spin} + 1;")
        emit(f"{pad}{E}seq_cycles = n * {spin};")
        emit(f"{pad}return (int)n;")
    elif loop:
        emit(f"{pad}const int head = {E}st_pc;")
        emit(f"{pad}{step_call}")
        closure = (f"{E}ram[(head + {loop['closure_offset']}) & 0xFFFF] == "
                   f"0x{loop['closure_value']:02X}")
        stable = "true"
        if loop["kind"] == "poll":
            # The polled value holds: plain memory, or a device read the
            # lease serves one fixed value.
            emit(f"{pad}const int polled = {E}shadow_read_target();")
            emit(f"{pad}const int lease = {E}lease_slot(polled);")
            stable = (f"(!{E}in_ranges({E}rwatch, polled) || "
                      f"(lease >= 0 && {E}lease_n[lease] == 0))")
        emit(f"{pad}long extra = 0;")
        emit(f"{pad}if ({E}st_pc == head && {closure} && {stable}) {{")
        emit(f"{pad}    const long per = {E}seq_cycles;  // one looping iteration")
        emit(f"{pad}    extra = (max_steps - i) / {steps} - 1;")
        emit(f"{pad}    if (max_cycles > 0) {{")
        emit(f"{pad}        const long room = max_cycles - cycles_this - {prefix} - 1;")
        emit(f"{pad}        if (extra > (room >= 0 ? room / per : 0)) extra = room >= 0 ? room / per : 0;")
        emit(f"{pad}    }}")
        if loop["kind"] == "count":
            reg = f"{E}st_{loop['register']}"
            delta = loop["delta"]
            left = reg if delta < 0 else f"256 - {reg}"
            emit(f"{pad}    if (extra > {left} - 1) extra = {left} - 1;  // the last falls through")
            emit(f"{pad}    if (extra > 0) {{")
            emit(f"{pad}        {reg} = ({reg} + {delta} * extra) & 0xFF;")
            emit(f"{pad}        {E}st_p = ({E}st_p & ~0x82) | ({reg} & 0x80) | ({reg} ? 0 : 0x02);")
            emit(f"{pad}    }}")
        emit(f"{pad}    if (extra < 0) extra = 0;")
        emit(f"{pad}    {E}seq_cycles = per * (1 + extra);")
        emit(f"{pad}}}")
        emit(f"{pad}return (int)((1 + extra) * {steps});")
    elif any(source == "model" for source, *_ in guards):
        # A superblock whose prediction failed declines: the next window
        # at this head (a shorter one) or single steps take over.
        emit(f"{pad}if ({step_call.rstrip(';')}) return {steps};")
    else:
        emit(f"{pad}{step_call}")
        emit(f"{pad}return {steps};")
    emit("            }")


def assemble_seq_module(lines, body, experts, core_digest, elem,
                         emit_step_core) -> tuple[str, int, bool]:
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
    emit("// shared machine state is reached through the typed reference E.")
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
    emit("")
    lines.extend(body)
    for j, exp in enumerate(experts):
        emit("")
        emit_step_core(f"step_core__{exp['name']}", f"__{exp['name']}",
                       exp["model"], exp["step_spec"], hook_id=SEQ_HOOK_IDS + j)
    # Window dispatch checks the head PC, bytes, interrupts, and budget.
    by_head: dict[int, list] = {}
    for exp in experts:
        by_head.setdefault(exp["head"], []).append(exp)
    for exps in by_head.values():  # longest first: a superblock before its prefix
        exps.sort(key=lambda exp: -exp["n_steps"])
    emit("")
    emit("    int dispatch(Eng& E, long i, long max_steps, long cycles_this,")
    emit("                 long max_cycles, bool stop_on_trap) {")
    emit("        if (E.nmi_pending || E.irq_ready() || stop_on_trap) return 0;")
    emit("        switch (E.st_pc) {")
    for head in sorted(by_head):
        emit(f"        case 0x{head:04X}:")
        for exp in by_head[head]:
            match = " && ".join(
                f"E.ram[0x{addr:04X}] == 0x{b:02X}" for addr, b in exp["match"]
            )
            if exp.get("has_stores"):
                # Licensed store targets stay below $2000; refuse
                # dispatch while any write watch reaches below it.
                match = f"E.watch_write_lo >= 0x2000 && {match}"
            emit_window_dispatch(emit, exp, "E.", match,
                                 f"step_core__{exp['name']}(E);")
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
    """Shared emission state: the union pass allocates memory targets and
    registers, expert passes resolve onto it, the epilogue
    emits from it.  Also the dialect: element type and member prefix
    ("" for the engine, "E." for a sequence module)."""

    def __init__(self, *, elem: str, E: str, ram_size: int = 0x10000,
                 count_reads: bool = False):
        # Count every attention-served memory read by address while the
        # engine's xray_on is set (the overlay's memory heatmap).
        self.count_reads = count_reads
        self.elem = elem
        self.E = E
        # Fused reads index a flat image of this many bytes.
        self.ram_size = ram_size
        self.n_memory_heads = 0
        # Running-argmax registers for constant-query heads, keyed by
        # (union lookup id, value pair): the union pass allocates in
        # head-scan order, expert passes resolve through fold provenance.
        self.reg_of: dict[tuple[int, int], int] = {}

    def xref(self, i):
        return f"x[{i}]"



def row_int(csr_m, r, ref, where):
    return expr_int(*row(csr_m, r), ref, where)


def row_raw(indices, data, ref, where):
    return expr_int(indices, data, ref, where)


def plan_model(ctx, mdl, label, union_lookup_of=None):
    """Per-layer emission plans for one model.

    The union pass (union_lookup_of is None) allocates the shared engine
    state -- the memory-structure targets and running-argmax registers keyed
    by (lookup id, value pair) -- in head-scan order. Expert passes resolve
    their heads onto that state through the fold's lookup provenance and
    never allocate.
    """
    is_union = union_lookup_of is None
    dm = mdl.tok.weight.shape[1]
    plans: list[dict] = []
    for li in range(len(mdl.attn)):
        check_ffn_kinds(mdl, li)
        # Layers are ragged: Q/K/V rows live in this layer's own head block.
        nh = mdl.attn[li].num_heads
        k_base = 2 * nh
        v_base = 4 * nh
        ip = csr(mdl.attn[li].in_proj_weight)
        op = csr(mdl.attn[li].out_proj.weight)
        fi = csr(mdl.ff_in[li].weight)
        fo = csr(mdl.ff_out[li].weight)
        ffn = fo.shape[1]
        kinds = mdl.head_kinds[li]
        ffn_kinds = mdl.ffn_kinds[li]
        tiebreak = mdl.head_tiebreak[li]
        mem_groups = mdl.head_memory_groups[li]
        lookup_ids = mdl.head_lookup_ids[li]
        pairs = mdl.head_pairs[li]
        memory_outputs = getattr(
            mdl, "head_memory_outputs",
            [[None] * a.num_heads for a in mdl.attn],
        )[li]

        lookup = [h for h in range(nh) if kinds[h] == 1]
        passthrough = [h for h in range(nh) if kinds[h] == 2]
        for h in lookup:
            if not tiebreak[h]:
                raise RuntimeError(
                    f"{label} L{li} head {h}: AVERAGE tie-break is not "
                    "supported by the exact integer attention structures"
                )

        for h in lookup:
            if mem_groups[h] >= 0 and is_union:
                ctx.n_memory_heads += 1

        # Structural passthrough/erase fusion: fold kind-2 heads and
        # neurons into direct D->D residual maps (no attention/ReGLU needed).
        pt_cols = [2 * h + c for h in passthrough for c in range(2)]
        attn_direct = compose(op, pt_cols, ip[[v_base + c for c in pt_cols], :])
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

        # ── Attention: licensed heads, then residual updates ────────
        for h in lookup:
            b = head_blocks.setdefault(h, [])
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
                    "structure-eligible"
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
                    k0_idx, k0_dat = row(ip, k_base + 2 * h)
                    k1_idx, k1_dat = row(ip, k_base + 2 * h + 1)
                    reg_score = integer_score_terms(
                        reg_q[0], reg_q[1],
                        list(zip(k0_dat, k0_idx, strict=True)),
                        list(zip(k1_dat, k1_idx, strict=True)),
                    )
                    if reg_score is None:
                        reg_q = None
                if reg_q is not None:
                    try:
                        v0_i = expr_int(*row(ip, v_base + 2 * h),
                                         ctx.xref, f"{label} L{li} reg v0")
                        v1_i = expr_int(*row(ip, v_base + 2 * h + 1),
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
                            f"(unsigned long long){qname} < {ctx.ram_size}ULL;",
                            *([f"        if ({ctx.E}xray_on && {stem}_valid) "
                               f"++{ctx.E}xray_reads[(int){qname}];"]
                              if ctx.count_reads else []),
                            f"        const long long {vname} = {stem}_valid ? "
                            f"{ctx.E}ram[(int){qname}] : 0;",
                            f"        const long long {aname} = "
                            f"{stem}_valid ? {qname} : 0;",
                        ))
                    _qname, vname, aname = names
                else:
                    b.append(f"            const long long qi = shr2({qi_expr});")
                    b.append("            long long mo[2] = {0, 0};")
                    # Group dispatch: qi is the address minus this group's
                    # base, so qi in [0, 4096) iff the address falls in this
                    # group.  Other groups' contributions are gate-zeroed
                    # downstream, and (0,0) is exactly the empty-group result.
                    b.append("            if ((unsigned long long)qi >> 12 == 0) {")
                    if ctx.count_reads:
                        b.append(f"                if ({ctx.E}xray_on) "
                                 f"++{ctx.E}xray_reads[({g} << 12) + (int)qi];")
                    b.append(f"                {ctx.E}mem_query({g}, qi, mo);")
                    b.append("            }")
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
                raise RuntimeError(
                    f"{label} L{li} head {h}: lookup is not served by a "
                    "supported exact attention structure"
                )
            b.append(f"        (void)ho{2 * h}; (void)ho{2 * h + 1};")

        # attn_direct reads pre-attention x; out_proj rows read the attention
        # outputs ho{..} and apply first.
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



def emit_layer(plan, suffix: str, signature: str) -> list[str]:
    """One layer's member function from its plan_model plan: licensed
    attention heads, the fused direct maps, then the ReGLU block."""
    li = plan["li"]
    out = [f"    void layer_{li}{suffix}({signature}) {{"]
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


def generate_source(model, step_spec: dict | None = None, experts=None,
                    seq_module_core: str | None = None,
                    mirror_cpu_ram: bool = False,
                    forward_hook: bool = False) -> tuple[str, int, bool]:
    """Return ``(source, memory_head_count, has_machine_step)``.

    ``step_spec`` enables the native machine protocol. Core ``experts`` are
    compact opcode or canonical shape execution models; sequence-module
    experts are licensed per-ROM regions. ``seq_module_core`` switches
    emission from the core engine to a module attached to that core.
    """
    experts = experts or []
    seq_mode = seq_module_core is not None
    if not seq_mode and any(
        e.get("kind") not in {"exec_opcode", "shape", "exec_int"} for e in experts
    ):
        raise RuntimeError(
            "core experts must be compact opcodes, shapes or interrupts"
        )
    if seq_mode and (
        not experts or any(e.get("kind") != "seq" for e in experts)
    ):
        raise RuntimeError("a sequence module takes kind='seq' experts only")
    if experts and step_spec is None:
        raise RuntimeError("execution models require a step_spec")

    # Shared-state prefix: "" emitting the engine itself (its own members),
    # "E." emitting a sequence module (a typed reference to the core engine).
    # Module-private scratch (xbuf) stays unprefixed.
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
    ctx = EmitterContext(elem=elem, E=E, count_reads=True)

    layer_plans = plan_model(ctx, model, "union")
    expert_plans = [
        (exp, plan_model(ctx, exp["model"], f"expert {exp['name']}",
                         exp["union_lookup_of"]))
        for exp in experts
    ]
    n_memory_heads = ctx.n_memory_heads
    n_regs = len(ctx.reg_of)
    layer_sig = f"{elem}* x"
    if seq_mode:
        layer_sig = f"Eng& E, {layer_sig}"

    # ── Emit the layer functions: one body per model -- the union stack
    # plus each rebuilt expert's own (smaller) stack.  Expert bodies are
    # complete emissions of their own weight matrices, not pruned copies of
    # the union's; the fold already removed the dead work at graph level. ──
    def emit_layer_fn(plan, suffix):
        return emit_layer(plan, suffix, layer_sig)

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

    def emit_step_core(fn_name, layer_suffix, mdl, spec, *, stage_writes=False,
                       trace=False, probe=False, hook_id=None):
        # One model forward over persistent st_* state. The union emits one
        # diagnostic receipt; execution models emit only final state, control
        # and ordered effects.  trace=True emits the same embedding and
        # layer stack writing the residual stream after the embedding and
        # after each layer to `snap`, and nothing else (the debugger's
        # dissection; run on a copy of the engine).  probe=True returns the
        # instruction's cycles and commits nothing (no interrupt may be
        # pending: the embed would take it).
        ins = spec["inputs"]
        sequence_final = spec.get("sequence_final")
        if stage_writes and sum(map(len, spec["sequence_writes"])) > 3:
            raise RuntimeError("compact execution exceeds three staged writes")
        if not sequence_final:
            hc, hv = spec["halt_head"]
            shc, shv = spec["state_head"]
        dm = mdl.tok.weight.shape[1]
        needs_x = (
            not sequence_final
            or bool(mdl.attn)
            or bool(sequence_final["model"])
            or any(spec["sequence_writes"])
            or spec["sequence_control"]["predicate"] is not None
            or spec["sequence_control"]["target_dimension"] is not None
            or spec["sequence_halt"] is not None
        )
        if not needs_x and (ins or spec["p_flags"]):
            raise RuntimeError("zero-work execution model retained live inputs")

        args = "Eng& E" if seq_mode else ""
        guards = [g for g in spec.get("sequence_guards", ()) if g[0] == "model"]
        if trace:
            args = f"{elem}* snap"
        elif not sequence_final and not probe:
            args += (", " if args else "") + "long long* out"
        returns = "long long" if probe else "bool" if guards else "void"
        emit(f"    {returns} {fn_name}({args}) {{")
        if needs_x:
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
        if trace:
            emit(f"        std::memcpy(snap, x, sizeof({elem}) * {dm});")
        # The forward hook (set_forward_hook): an outside runtime runs this
        # expert's layers on the same residual stream, `hook_id` naming it.
        hooked = hook_id is not None and needs_x
        if hooked:
            flag = "Eng::FORWARD_HOOK" if seq_mode else "FORWARD_HOOK"
            emit(f"        if (!({flag} && {E}forward_hook && "
                 f"{E}forward_hook({E}forward_hook_ctx, {hook_id}, x))) {{")
        for li in range(len(mdl.attn)):
            emit(f"        layer_{li}{layer_suffix}("
                 f"{'E, ' if seq_mode else ''}x);")
            if trace:
                emit(f"        std::memcpy(snap + {(li + 1) * dm}, x, sizeof({elem}) * {dm});")
        if hooked:
            emit("        }")
        if trace:
            emit("    }")
            return
        if probe:
            emit(f"        --{E}seq;")
            emit(f"        return {rdv(spec['outputs']['cycles'])} & 0xFF;")
            emit("    }")
            return
        if sequence_final:
            if guards:
                # A superblock's predicted branches: unless the model's
                # flags agree, nothing commits and the dispatch declines.
                held = " && ".join(
                    f"(({rdv(slot)} & 1) != 0) == {'true' if expected else 'false'}"
                    for _source, slot, expected in guards)
                emit(f"        if (!({held})) {{ --{E}seq; return false; }}")
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
                    op = "|=" if constants[name] else "&="
                    value = f"0x{mask:X}" if constants[name] else f"~0x{mask:X}"
                    emit(f"        {E}st_p {op} {value};")
                    continue
                else:
                    continue
                emit(f"        {E}st_p = ({E}st_p & ~0x{mask:X}) | ({value});")
            control = spec["sequence_control"]
            if control["target_dimension"] is not None:
                target = rdv(control["target_dimension"])
                fallthrough = target
            elif control["absolute_operand_offset"] is not None:
                offset = control["absolute_operand_offset"]
                target = (
                    f"({E}ram[({E}st_pc + {offset}) & 0xFFFF] | "
                    f"({E}ram[({E}st_pc + {offset + 1}) & 0xFFFF] << 8))"
                )
                fallthrough = target
            elif control["relative_to_head"]:
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
            cycles = spec["sequence_cycles"]
            emit(f"        {E}seq_cycles = " + " + ".join(
                [str(cycles["constant"])]
                + [f"({rdv(slot)} & 0xFF)" for slot in cycles["model"]]
            ) + ";")
            if spec["sequence_halt"] is not None:
                emit(f"        {E}st_halted = "
                     f"({rdv(spec['sequence_halt'])} & 1) != 0;")
            for step_writes in spec["sequence_writes"]:
                for addr_slot, value_slot in step_writes:
                    if stage_writes:
                        emit(f"        exec_write_addr[exec_n_writes] = "
                             f"(int)({rdv(addr_slot)} & 0xFFFF);")
                        emit(f"        exec_write_value[exec_n_writes] = "
                             f"(int)({rdv(value_slot)} & 0xFF);")
                        emit("        ++exec_n_writes;")
                    else:
                        emit(f"        {E}insert_write((int)({rdv(addr_slot)} & 0xFFFF), "
                             f"(int)({rdv(value_slot)} & 0xFF), {E}seq + 1);")
            if guards:
                emit("        return true;")
            emit("    }")
            return
        outs = spec["outputs"]
        for i, (name, mask) in enumerate((
            ("current_pc", 0xFFFF), ("new_a", 0xFF), ("new_x", 0xFF),
            ("new_y", 0xFF), ("new_s", 0xFF), ("new_p", 0xFF),
            ("fetched_opcode", 0xFF), ("fetched_operand_lo", 0xFF),
            ("fetched_operand_hi", 0xFF), ("ea", 0xFFFF), ("data_out", 0xFF),
            ("new_halted", 0xFF), ("next_pc", 0xFFFF),
        )):
            emit(f"        out[{i}] = {rdv(outs[name])} & 0x{mask:X};")
        emit("        { long long n_writes = 0;")
        for store, addr, value in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            emit(f"        if (({rdv(outs[store])} & 0xFF) != 0) {{")
            emit(f"            out[15 + 2 * n_writes] = {rdv(outs[addr])} & 0xFFFF;")
            emit(f"            out[16 + 2 * n_writes] = {rdv(outs[value])} & 0xFF;")
            emit("            ++n_writes;")
            emit("        }")
        emit("        out[14] = n_writes;")
        emit(f"        out[21] = {rdv(outs['cycles'])} & 0xFF;")
        emit("        for (long long i = 0; i < n_writes; ++i)")
        emit(f"            {E}insert_write((int)out[15 + 2 * i], "
             f"(int)out[16 + 2 * i], {E}seq + 1);")
        emit("        }")
        emit(f"        const long long halt_logit = "
             f"{expr_int(hc, hv, lambda i: f'x[{i}]', 'halt head')};")
        emit(f"        const long long state_logit = "
             f"{expr_int(shc, shv, lambda i: f'x[{i}]', 'state head')};")
        emit("        out[13] = halt_logit > state_logit ? 1 : 0;")
        emit("        // Advance the architectural state (inputs are fully")
        emit("        // consumed by the embed above before this update).")
        emit(f"        {E}st_pc = (int)out[12];")
        emit(f"        {E}st_a = (int)out[1]; {E}st_x = (int)out[2]; "
             f"{E}st_y = (int)out[3];")
        emit(f"        {E}st_s = (int)out[4]; {E}st_p = (int)out[5];")
        emit(f"        {E}st_ir = (int)out[6]; {E}st_oplo = (int)out[7]; "
             f"{E}st_ophi = (int)out[8];")
        emit(f"        {E}st_ea = (int)out[9]; {E}st_data = (int)out[10];")
        emit(f"        {E}st_halted = out[11] != 0;")
        emit("    }")

    if seq_mode:
        return assemble_seq_module(
            lines, body, experts, seq_module_core, elem,
            emit_step_core,
        )

    # The C++ run loop (KernelBus shape) needs the same memory-write path the
    # machine_step uses, so it is emitted under the same condition.
    has_run = step_spec is not None and n_memory_heads > 0

    emit("// Generated exact-int32 engine (model/engine_gen.py).")
    emit(INCLUDES)
    emit("")
    emit("struct ENGINE_T {")
    emit(f"    static constexpr bool MIRROR_CPU_RAM = {'true' if mirror_cpu_ram else 'false'};")
    # Built for an outside forward runtime (set_forward_hook): only then do
    # expert steps branch on the hook -- an opaque call there costs the
    # compiled engine its register-resident state.
    emit(f"    static constexpr bool FORWARD_HOOK = {'true' if forward_hook else 'false'};")
    emit(f"    static constexpr int D = {d_model};")
    emit("    long seq = -1;")
    emit(f"    static constexpr int F = {f_bits};")
    emit("    bool xray_on = false;  // count memory reads (xray_reads)")
    emit("    unsigned int xray_reads[65536] = {};")
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
    emit("        for (int g = 0; g < 16; ++g)")
    emit("            for (int w = 0; w < 64; ++w) mem_bits[g][w] = 0ULL;")
    if n_regs:
        emit(f"        for (int r = 0; r < {n_regs}; ++r) reg_any[r] = false;")
    if has_run:
        emit("        for (int i = 0; i < 65536; ++i) ram[i] = 0;")
        emit("        std::memset(shape_of_pc, 0, sizeof(shape_of_pc));")
        emit("        st_pc = 0; st_a = st_x = st_y = 0; st_s = 0xFD; st_p = 0x24;")
        emit("        st_ir = st_oplo = st_ophi = st_ea = st_data = 0;")
        emit("        st_halted = false; nmi_pending = false; irq_pending = false;")
        emit("        resume_read_skip = false;")
        emit("        clear_read_lease();")
        emit("        step_index = 0; cycles_total = 0;")
        emit("        clear_watches();")
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
    emit("    void insert_write(int addr, int value, long pos) {")
    emit("        ++seq;")
    emit("        if (seq != pos)")
    emit('            throw std::runtime_error("direct WRITE position does not match engine sequence");')
    emit("        addr &= 0xFFFF; value &= 0xFF;")
    emit("        const int first = (MIRROR_CPU_RAM && addr < 0x2000) ? (addr & 0x07FF) : addr;")
    emit("        const int last = (MIRROR_CPU_RAM && addr < 0x2000) ? first + 0x1800 : first;")
    emit("        for (int mapped = first; mapped <= last; mapped += 0x0800) {")
    if has_run:
        emit("            ram[mapped] = (unsigned char)value;  // shadow-decode mirror")
    emit("            const int group = (mapped >> 12) & 0xF;")
    emit("            const int wl = mapped & 0xFFF;")
    emit("            mem_val[group][wl] = value;")
    emit("            mem_seq[group][wl] = pos;")
    emit("            mem_bits[group][wl >> 6] |= 1ULL << (wl & 63);")
    emit("        }")
    emit("    }")

    has_step = has_run
    if has_step:
        emit("")
        emit("    // One whole 6502 instruction from the persistent st_* state:")
        emit("    // STATE embed -> layers -> readout -> direct WRITE inserts ->")
        emit("    // halt logits.  Fills the 22-slot out buffer (trace row, next")
        emit("    // state, writes, cycles) and advances st_* to the next instruction.")
        emit("")
        emit_step_core("step_core", "", model, step_spec)
        emit("")
        emit_step_core("trace_core", "", model, step_spec, trace=True)
        emit("")
        emit_step_core("union_cycles", "", model, step_spec, probe=True)
        for j, exp in enumerate(experts):
            emit("")
            emit_step_core(f"step_core__{exp['name']}", f"__{exp['name']}",
                           exp["model"], exp["step_spec"],
                           stage_writes=exp.get("kind")
                           in ("exec_opcode", "exec_int"),
                           hook_id=j)

        # Compact opcode execution is selected by the byte at st_pc. Pending
        # interrupts and tracing bypass it through step_core's union receipt.
        exec_op_experts = [
            e for e in experts if e.get("kind") == "exec_opcode"
        ]
        shape_experts = [e for e in experts if e.get("kind") == "shape"]
        emit("")
        emit("    int exec_n_writes = 0;")
        emit("    int exec_write_addr[3], exec_write_value[3];")
        # Compact per-opcode dispatch; an opcode with no case returns false
        # and the run loop takes the union step_core.
        emit("    bool execution_step(unsigned char opcode) {")
        emit("        exec_n_writes = 0;")
        if exec_op_experts:
            emit("        switch (opcode) {")
            for exp in exec_op_experts:
                emit(f"            case 0x{exp['byte']:02X}: "
                     f"step_core__{exp['name']}(); return true;")
            emit("        }")
        emit("        return false;")
        emit("    }")
        emit("")
        emit("    // The next instruction's cycles without running it (a watched")
        emit("    // read's stamp): its expert's forward on saved state, else the")
        emit("    // union's.  No interrupt is pending when a read pauses.")
        emit("    long probe_cycles() {")
        emit("        const auto saved = std::tuple{st_pc, st_a, st_x, st_y, st_s, st_p, st_halted, seq};")
        emit("        const bool ran = execution_step(ram[st_pc]);")
        emit("        std::tie(st_pc, st_a, st_x, st_y, st_s, st_p, st_halted, seq) = saved;")
        emit("        return ran ? seq_cycles : union_cycles();")
        emit("    }")

        int_experts = {e["nmi"]: e for e in experts
                       if e.get("kind") == "exec_int"}
        emit("")
        emit("    bool interrupt_step(bool nmi) {")
        emit("        exec_n_writes = 0;")
        for is_nmi in (True, False):
            exp = int_experts.get(is_nmi)
            if exp is None:
                continue
            cond = "nmi" if is_nmi else "!nmi"
            emit(f"        if ({cond}) {{ step_core__{exp['name']}();"
                 f" return true; }}")
        emit("        return false;")
        emit("    }")

        # Canonical shape windows are compiled once and revalidated against RAM.
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
                if exp["spin"]:
                    checks += (" && ram[st_pc + 1] == (st_pc & 0xFF)"
                               " && ram[st_pc + 2] == ((st_pc >> 8) & 0xFF)")
                emit(f"        case {j + 1}:  // {exp['name']}")
                emit_window_dispatch(emit, exp, "", checks,
                                     f"step_core__{exp['name']}();")
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
        emit('        if ((int)out_arr.size() != 22) throw std::runtime_error("step out buffer must have 22 slots");')
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
        # The dissection: one union forward from the current machine state
        # on a copy of the engine, every layer's residual stream returned
        # as fixed-point ints (rows: embedding, then each layer).
        # The run's profile since the last call: block entries, and each
        # conditional branch's runs and taken runs (rows 0, 1, 2).
        emit("        .def(\"take_profile\", [](ENGINE_T& e) {")
        emit("            py::array_t<long> out({3, 65536});")
        emit("            long* o = out.mutable_data();")
        emit("            for (int i = 0; i < 65536; ++i) {")
        emit("                o[i] = std::exchange(e.block_entries[i], 0);")
        emit("                o[65536 + i] = std::exchange(e.branch_runs[i], 0);")
        emit("                o[131072 + i] = std::exchange(e.branch_taken[i], 0);")
        emit("            }")
        emit("            return out;")
        emit("        })")
        emit("        .def(\"xray_take\", [](ENGINE_T& e) {")
        emit("            py::list paths;")
        emit("            for (long& n : e.xray_paths) paths.append(std::exchange(n, 0));")
        emit("            py::list dispatches;")
        emit("            for (long& n : e.xray_dispatches) dispatches.append(std::exchange(n, 0));")
        emit("            py::array_t<long> ops(256);")
        emit("            for (int i = 0; i < 256; ++i) ops.mutable_data()[i] = std::exchange(e.xray_opcodes[i], 0);")
        emit("            return py::make_tuple(paths, ops, dispatches);")
        emit("        })")
        # Reads per address since the last call (then zeroed); `on`
        # keeps counting enabled.
        emit("        .def(\"xray_reads\", [](ENGINE_T& e, bool on) {")
        emit("            py::array_t<unsigned int> out(65536);")
        emit("            std::memcpy(out.mutable_data(), e.xray_reads, sizeof e.xray_reads);")
        emit("            std::memset(e.xray_reads, 0, sizeof e.xray_reads);")
        emit("            e.xray_on = on;")
        emit("            return out;")
        emit("        })")
        emit("        .def(\"trace_union\", [](const ENGINE_T& e) {")
        emit("            auto copy = std::make_unique<ENGINE_T>(e);")
        emit(f"            py::array_t<int> out({{{n_layers + 1}, {d_model}}});")
        emit("            copy->trace_core(out.mutable_data());")
        emit("            return py::make_tuple(out, ENGINE_T::F);")
        emit("        })")
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
    machine_abi: Callable
    watch_read: Callable
    watch_write: Callable
    clear_watches: Callable
    set_nmi: Callable
    set_irq: Callable
    clear_irq: Callable
    seed_image: Callable
    map_region: Callable
    read_ram: Callable
    dump_ram: Callable
    attach_seq: Callable
    detach_seq: Callable
    add_window: Callable
    clear_windows: Callable
    digest: str

    def __init__(self, model, *, step_spec: dict | None = None,
                 experts=None, cache_key: str | None = None,
                 shapes_meta=None, mirror_cpu_ram: bool = False,
                 forward_hook: bool = False):
        self.n_experts = len(experts or [])
        self.mirror_cpu_ram = bool(mirror_cpu_ram)
        self.forward_hook = bool(forward_hook)
        # Shape-window scan metadata (id/signature/length/close rule) so a
        # cache-hit load can populate the runtime table without refolding.
        self.shapes_meta = list(shapes_meta or [])
        source, self.n_memory_heads, self.has_machine_step = generate_source(
            model, step_spec, experts=experts,
            mirror_cpu_ram=self.mirror_cpu_ram, forward_hook=self.forward_hook,
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
        module = load_engine_module(digest)
        if cache_key is not None:
            manifest_put(cache_key, {
                "digest": digest, "n_experts": self.n_experts,
                "n_memory_heads": self.n_memory_heads,
                "has_machine_step": self.has_machine_step,
                "mirror_cpu_ram": self.mirror_cpu_ram,
                "forward_hook": self.forward_hook,
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
        self.mirror_cpu_ram = bool(entry.get("mirror_cpu_ram", False))
        self.forward_hook = bool(entry.get("forward_hook", False))
        self.shapes_meta = entry.get("shapes") or []
        module = load_engine_module(entry["digest"])
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
            self.set_nmi = self.engine.set_nmi
            self.set_irq = self.engine.set_irq
            self.clear_irq = self.engine.clear_irq
            self.seed_image = self.engine.seed_image
            self.map_region = self.engine.map_region
            self.read_ram = self.engine.read_ram
            self.dump_ram = self.engine.dump_ram
            self.attach_seq = self.engine.attach_seq
            self.detach_seq = self.engine.detach_seq
            self.add_window = self.engine.add_window
            self.clear_windows = self.engine.clear_windows
            self.machine_abi = self.engine.machine_abi

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
        module = load_seqmod_module(digest)
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
        module = load_seqmod_module(entry["digest"])
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
