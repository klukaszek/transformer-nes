"""C++ kernel backend for the 6502 graph evaluator.

Emits the same live schedule that evaluator.py compiles to Python, but as a
single C++ translation unit that owns the hull attention heads, the token
insert paths, and the autoregressive run loop.  The Python evaluator remains
the exact reference; this backend exists for throughput (the NES path needs
orders of magnitude more steps than CPython can deliver).

Numerics are intended to be bit-identical to the Python evaluator: same IEEE
double operations, in the same order, against the same hull code
(hull2d_cht.h) with the same tie-break configuration.
"""

from __future__ import annotations

import hashlib
import os
import time

from transformer_vm.cpu6502.evaluator import (
    _OP_LOOKUP,
    _OP_PERSIST,
    _OP_REGLU,
    _RUNTIME_READ_DIMS,
    Cpu6502Runtime,
)
from transformer_vm.cpu6502.run_protocol import cycle_table, read_mode_table
from transformer_vm.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_vm.graph.core import PersistDimension
from transformer_vm.model.classify import memory_query, register_query

_ATTENTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_kernel_cache")

# STATE-token override fields, in the order the C++ insert_state() takes them.
_STATE_FIELDS = (
    "is_state", "in_pc_lo", "in_pc_hi", "in_a", "in_x", "in_y", "in_s",
    "in_p_c", "in_p_z", "in_p_i", "in_p_d", "in_p_b", "in_p_u", "in_p_v", "in_p_n",
    "in_ir", "in_oplo", "in_ophi", "in_ea_lo", "in_ea_hi", "in_data", "in_halted",
    "in_nmi",
)

def _c_num(coeff) -> str:
    if isinstance(coeff, int):
        return f"{coeff}.0"
    return repr(float(coeff))


def _c_expr(arg, ref) -> str:
    """Render a compiled (n, ids, coeffs) expression as C source.

    ref(dim_id) returns the C lvalue for a dimension.  Term order matches
    the Python emitter so float accumulation is identical.
    """
    n, ids, coeffs = arg
    if n == 0:
        return "0.0"
    parts = []
    for dim_id, coeff in zip(ids, coeffs, strict=True):
        r = ref(dim_id)
        if coeff == 1:
            parts.append(r)
        elif coeff == -1:
            parts.append(f"-{r}")
        else:
            parts.append(f"{_c_num(coeff)}*{r}")
    return " + ".join(parts)


class _KernelSpec:
    """Everything the C++ emitter needs, extracted from a Cpu6502Runtime."""

    def __init__(self, rt: Cpu6502Runtime):
        self.rt = rt
        self.ndims = rt.num_dims
        self.computed = {dim_id for dim_id, _op, _arg in rt._live_schedule}

        # Unique cache groups (memory lookups share one backing cache).
        cache_to_group: dict[int, int] = {}
        self.groups: list[dict] = []
        self.lu_group: dict[int, int] = {}
        for lu in rt.all_lookups:
            attn = rt.attention[lu.id]
            key = id(attn.cache)
            if key not in cache_to_group:
                plan = rt._insert_kv_plan(lu)
                if plan is None:
                    raise NotImplementedError(
                        f"lookup {lu.id} needs the generic insert path; C++ kernel unsupported"
                    )
                cache_to_group[key] = len(self.groups)
                self.groups.append({
                    "rep": lu,
                    "nvp": attn.num_value_pairs,
                    "nv": attn.nv,
                    "latest": lu.tie_break == "latest",
                    "plan": plan,
                })
            self.lu_group[lu.id] = cache_to_group[key]

        self.start_group = self.lu_group[rt._start_lu.id]
        self.state_group = self.lu_group[rt._state_lu.id]
        self.memory_groups = sorted({self.lu_group[lu.id] for lu in rt._memory_read_lus})

        def input_id(name):
            return getattr(rt, name).id

        self.builtin_ids = {
            "one": rt._one.id,
            "position": rt._position.id,
            "position_sq": rt._position_sq.id,
            "inv_log_pos": rt._inv_log_pos.id,
        }
        self.write_ids = {name: input_id(name) for name in
                          ("byte_value", "is_memory_write", "addr_lo", "addr_hi")}
        self.start_ids = {name: input_id(name) for name in
                          ("start_pc_lo", "start_pc_hi", "is_start")}
        self.state_ids = {name: input_id(name) for name in _STATE_FIELDS}

        # Every embedding coefficient must be shadowed by an override the
        # kernel sets, otherwise the token builders below would be wrong.
        overrides = {
            "WRITE": set(self.write_ids.values()),
            "start": set(self.start_ids.values()),
            "STATE": set(self.state_ids.values()),
        }
        for token, allowed in overrides.items():
            for dim in rt.input_tokens[token].terms:
                if dim.id not in allowed:
                    raise NotImplementedError(
                        f"{token} embedding term {dim!r} is not overridden by the kernel"
                    )

        self.trace_ids = {name: rt.dim_id_by_name[name] for name in _RUNTIME_READ_DIMS}
        self.outputs = rt._indexed_outputs  # [(name, expr)] in score order

        # ── O(1) attention-state classification (mirrors engine_gen) ──
        # Memory lookups whose query is a positive scalar multiple of
        # [local_address, 1] and whose values are exactly (byte_value, addr)
        # are served by the exact nearest-written-address structure; LATEST
        # lookups with constant [qx, qy] queries by running-argmax registers
        # folded at insert time.  Anything else keeps its hull -- correctness
        # never rests on the classification, only speed.
        one_id = rt._one.id
        pos_ids = {rt._position.id, rt._position_sq.id, rt._inv_log_pos.id}

        def q_terms(cq):
            _n, ids, coeffs = cq
            return list(zip(coeffs, ids, strict=True))

        self.mem_lus: dict[int, dict] = {}
        self.regs: list[dict] = []
        self.lu_reg: dict[int, int] = {}
        reg_of: dict[tuple, int] = {}
        self.hull_groups: set[int] = set()
        mem_ids = {lu.id for lu in rt._memory_read_lus}
        for lu in rt.all_lookups:
            gi = self.lu_group[lu.id]
            grp = self.groups[gi]
            qx_c, qy_c = rt._lookup_query_exprs[lu.id]
            if lu.id in mem_ids:
                mg = getattr(lu, "memory_group", None)
                terms = None
                if (
                    mg is not None and grp["nvp"] == 1
                    and lu.tie_break == "latest"
                    and {d.name: c for d, c in lu.value_exprs[0].terms.items()}
                    == {"byte_value": 1}
                    and {d.name: c for d, c in lu.value_exprs[1].terms.items()}
                    == {"addr_lo": 1, "addr_hi": 256}
                ):
                    terms = memory_query(q_terms(qx_c), q_terms(qy_c),
                                         one_id=one_id, banned_ids=pos_ids)
                if terms is not None:
                    self.mem_lus[lu.id] = {"g": int(mg), "terms": terms}
                else:
                    self.hull_groups.add(gi)
                continue
            reg_q = register_query(q_terms(qx_c), q_terms(qy_c),
                                   one_id=one_id, latest=grp["latest"])
            if reg_q is not None:
                qx, qy = reg_q
                key = (gi, qx, qy)
                ri = reg_of.get(key)
                if ri is None:
                    ri = reg_of[key] = len(self.regs)
                    self.regs.append({"gi": gi, "qx": qx, "qy": qy,
                                      "nv": grp["nvp"] * 2})
                self.lu_reg[lu.id] = ri
            else:
                self.hull_groups.add(gi)

        self.grp_regs: dict[int, list[int]] = {}
        for ri, reg in enumerate(self.regs):
            self.grp_regs.setdefault(reg["gi"], []).append(ri)
        # Groups whose insert function is still needed: hull readers plus
        # register folds.  Fully structure-served memory groups drop out.
        self.live_groups = self.hull_groups | set(self.grp_regs)


def generate_cpp(rt: Cpu6502Runtime) -> str:
    spec = _KernelSpec(rt)
    L: list[str] = []
    emit = L.append

    def gref(dim_id):
        return f"g[{dim_id}]"

    emit('#include <pybind11/pybind11.h>')
    emit('#include <pybind11/numpy.h>')
    emit('#include <cmath>')
    emit('#include <cstring>')
    emit('#include <utility>')
    emit('#include <vector>')
    emit('#include "hull2d_cht.h"')
    emit('')
    emit('namespace py = pybind11;')
    emit('')
    emit(f'constexpr int NDIMS = {spec.ndims};')
    emit('')
    # Python's round() is half-to-even, and the graph genuinely produces
    # half-integer flag values (e.g. carry=0.5 from the +0.5 threshold
    # tricks) whose correct resolution depends on it.  nearbyint uses the
    # default FE_TONEAREST mode, which is also half-to-even.
    emit('static inline long iround(double x) { return (long)nearbyint(x); }')
    emit('')
    # KERNEL_T is replaced with a hash-unique struct name before compiling:
    # the bcd=True and bcd=False kernels have different layouts, and two
    # dylibs defining a same-named struct is an ODR violation (their RTTI
    # gets coalesced by the dynamic linker, corrupting pybind dispatch).
    emit('struct KERNEL_T {')

    # ── Members ────────────────────────────────────────────────────
    for gi, grp in enumerate(spec.groups):
        if gi in spec.hull_groups:
            emit(f'    HardAttentionHead grp{gi}[{grp["nvp"]}];')
            emit(f'    int grp{gi}_seq = -1;')
    emit('    double g[NDIMS] = {0};')
    emit('    std::vector<int> g_set;')
    emit('')
    emit('    // Exact memory-head argmax: nearest address wins; equal-distance')
    emit('    // ties use the latest write. State is O(1) in the token log.')
    emit('    int mem_val[16][4096];')
    emit('    long long mem_seq[16][4096];')
    emit('    unsigned long long mem_bits[16][64];')
    emit('')
    emit('    void mem_query(const int g_, const long long q, double out[2]) const {')
    emit('        const unsigned long long* bits = mem_bits[g_];')
    emit('        if (q >= 0 && q <= 4095 && (bits[q >> 6] >> (q & 63)) & 1ULL) {')
    emit('            out[0] = (double)mem_val[g_][q];')
    emit('            out[1] = (double)(((long long)g_ << 12) + q);')
    emit('            return;')
    emit('        }')
    emit('        long long below = -1, above = -1;')
    emit('        if (q >= 0) {')
    emit('            const long long start = q < 4095 ? q : 4095;')
    emit('            int w = (int)(start >> 6);')
    emit('            unsigned long long m = bits[w] & (~0ULL >> (63 - (start & 63)));')
    emit('            while (true) {')
    emit('                if (m) { below = ((long long)w << 6) + (63 - __builtin_clzll(m)); break; }')
    emit('                if (--w < 0) break;')
    emit('                m = bits[w];')
    emit('            }')
    emit('        }')
    emit('        if (q <= 4095) {')
    emit('            const long long start = q > 0 ? q : 0;')
    emit('            int w = (int)(start >> 6);')
    emit('            unsigned long long m = bits[w] & (~0ULL << (start & 63));')
    emit('            while (true) {')
    emit('                if (m) { above = ((long long)w << 6) + __builtin_ctzll(m); break; }')
    emit('                if (++w > 63) break;')
    emit('                m = bits[w];')
    emit('            }')
    emit('        }')
    emit('        long long win;')
    emit('        if (below < 0 && above < 0) { out[0] = 0.0; out[1] = 0.0; return; }')
    emit('        else if (below < 0) win = above;')
    emit('        else if (above < 0) win = below;')
    emit('        else {')
    emit('            const long long db = q - below, da = above - q;')
    emit('            if (db < da) win = below;')
    emit('            else if (da < db) win = above;')
    emit('            else win = mem_seq[g_][below] > mem_seq[g_][above] ? below : above;')
    emit('        }')
    emit('        out[0] = (double)mem_val[g_][win];')
    emit('        out[1] = (double)(((long long)g_ << 12) + win);')
    emit('    }')
    emit('')
    if spec.regs:
        emit('    // Running-argmax registers for constant-query LATEST lookups')
        emit('    // (start/state): the score of each token is a fixed function of')
        emit('    // its key, so the argmax folds at insert time and a query is a')
        emit('    // register read.  The state log stops growing per instruction.')
        for ri, reg in enumerate(spec.regs):
            emit(f'    double reg{ri}_s; double reg{ri}_v[{reg["nv"]}]; '
                 f'bool reg{ri}_any = false;')
        emit('')
    emit('    void reset() {')
    for gi, grp in enumerate(spec.groups):
        if gi in spec.hull_groups:
            emit(f'        for (int h = 0; h < {grp["nvp"]}; ++h) grp{gi}[h].clear();')
            emit(f'        grp{gi}_seq = -1;')
    emit('        for (int g_ = 0; g_ < 16; ++g_)')
    emit('            for (int w = 0; w < 64; ++w) mem_bits[g_][w] = 0ULL;')
    for ri in range(len(spec.regs)):
        emit(f'        reg{ri}_any = false;')
    emit('        for (int id : g_set) g[id] = 0.0;')
    emit('        g_set.clear();')
    emit('    }')
    emit('')
    emit('    inline void setg(int id, double v) { g[id] = v; g_set.push_back(id); }')
    emit('')
    emit('    inline void new_token(double pos) {')
    emit('        for (int id : g_set) g[id] = 0.0;')
    emit('        g_set.clear();')
    emit(f'        setg({spec.builtin_ids["one"]}, 1.0);')
    emit(f'        setg({spec.builtin_ids["position"]}, pos);')
    emit(f'        setg({spec.builtin_ids["position_sq"]}, pos * pos);')
    emit(f'        setg({spec.builtin_ids["inv_log_pos"]}, 1.0 / std::log(2.0) - 1.0 / std::log(pos + 2.0));')
    emit('    }')

    # ── Per-group insert functions ─────────────────────────────────
    # Emitted only for groups something still reads through: hull readers
    # and register folds.  Fully structure-served memory groups drop out
    # (insert_write updates the structure directly from (addr, value)).
    compile_expr = Cpu6502Runtime._compile_expr
    for gi, grp in enumerate(spec.groups):
        if gi not in spec.live_groups:
            continue
        lu = grp["rep"]
        order, input_ids = grp["plan"]
        emit('')
        emit(f'    void grp{gi}_insert() {{')
        local_ids = set()

        def dref(dim_id, local_ids=local_ids):
            return f"d{dim_id}" if dim_id in local_ids else gref(dim_id)

        for dim in order:
            if isinstance(dim, PersistDimension):
                emit(f'        const double d{dim.id} = {_c_expr(compile_expr(dim.expr), dref)};')
            else:
                b_src = _c_expr(compile_expr(dim.b_expr), dref)
                a_src = _c_expr(compile_expr(dim.a_expr), dref)
                emit(f'        const double _b{dim.id} = {b_src};')
                emit(f'        const double d{dim.id} = _b{dim.id} > 0.0 ? ({a_src}) * _b{dim.id} : 0.0;')
            local_ids.add(dim.id)
        kx = _c_expr(compile_expr(lu.key_exprs_2d[0]), dref)
        ky = _c_expr(compile_expr(lu.key_exprs_2d[1]), dref)
        emit(f'        const double key[2] = {{ {kx}, {ky} }};')
        for p in range(grp["nvp"]):
            for slot in range(2):
                vi = p * 2 + slot
                src = _c_expr(compile_expr(lu.value_exprs[vi]), dref) if vi < grp["nv"] else "0.0"
                emit(f'        const double val{p}_{slot} = {src};')
        if gi in spec.hull_groups:
            emit(f'        ++grp{gi}_seq;')
            for p in range(grp["nvp"]):
                emit(f'        {{ const double val[2] = {{ val{p}_0, val{p}_1 }}; '
                     f'grp{gi}[{p}].insert(key, val, grp{gi}_seq); }}')
        for ri in spec.grp_regs.get(gi, ()):
            reg = spec.regs[ri]
            qx, qy = _c_num(reg["qx"]), _c_num(reg["qy"])
            # clear_key-gated tokens (ky -= 1e30) can never win a query;
            # replace-on->= reproduces the LATEST tie-break exactly.
            emit('        if (key[1] > -1.0e29) {')
            emit(f'            const double s_{ri} = {qx}*key[0] + {qy}*key[1];')
            emit(f'            if (!reg{ri}_any || s_{ri} >= reg{ri}_s) {{')
            emit(f'                reg{ri}_any = true; reg{ri}_s = s_{ri};')
            for k in range(reg["nv"]):
                emit(f'                reg{ri}_v[{k}] = val{k // 2}_{k % 2};')
            emit('            }')
            emit('        }')
        emit('    }')

    # ── Token builders ─────────────────────────────────────────────
    emit('')
    emit('    void insert_start(int pc, double pos) {')
    emit('        new_token(pos);')
    emit(f'        setg({spec.start_ids["start_pc_lo"]}, (double)(pc & 0xFF));')
    emit(f'        setg({spec.start_ids["start_pc_hi"]}, (double)((pc >> 8) & 0xFF));')
    emit(f'        setg({spec.start_ids["is_start"]}, 1.0);')
    emit(f'        grp{spec.start_group}_insert();')
    emit('    }')
    emit('')
    emit('    void insert_write(int addr, int value, double pos) {')
    emit('        const int wg = (addr >> 12) & 0xF, wl = addr & 0xFFF;')
    emit('        mem_val[wg][wl] = value & 0xFF;')
    emit('        mem_seq[wg][wl] = (long long)pos;')
    emit('        mem_bits[wg][wl >> 6] |= 1ULL << (wl & 63);')
    hull_mem = [gi for gi in spec.memory_groups if gi in spec.live_groups]
    if hull_mem:
        emit('        new_token(pos);')
        emit(f'        setg({spec.write_ids["byte_value"]}, (double)value);')
        emit(f'        setg({spec.write_ids["is_memory_write"]}, 1.0);')
        emit(f'        setg({spec.write_ids["addr_lo"]}, (double)(addr & 0xFF));')
        emit(f'        setg({spec.write_ids["addr_hi"]}, (double)((addr >> 8) & 0xFF));')
        for gi in hull_mem:
            emit(f'        grp{gi}_insert();')
    emit('    }')
    emit('')
    state_args = ", ".join(f"double {name}" for name in _STATE_FIELDS[1:])
    emit(f'    void insert_state(double pos, {state_args}) {{')
    emit('        new_token(pos);')
    emit(f'        setg({spec.state_ids["is_state"]}, 1.0);')
    for name in _STATE_FIELDS[1:]:
        emit(f'        setg({spec.state_ids[name]}, {name});')
    emit(f'        grp{spec.state_group}_insert();')
    emit('    }')

    # ── Step function ──────────────────────────────────────────────
    emit('')
    emit('    void step(double* v) {')
    step_locals = set(spec.rt._step_input_ids)

    def sref(dim_id):
        return f"d{dim_id}" if dim_id in step_locals else gref(dim_id)

    for dim_id in spec.rt._step_input_ids:
        emit(f'        const double d{dim_id} = g[{dim_id}];')
    emitted_lookups = set()
    for dim_id, op, arg in spec.rt._live_schedule:
        if op == _OP_PERSIST:
            emit(f'        const double d{dim_id} = {_c_expr(arg, sref)};')
            step_locals.add(dim_id)
        elif op == _OP_REGLU:
            a_expr, b_expr = arg
            emit(f'        const double _b{dim_id} = {_c_expr(b_expr, sref)};')
            emit(f'        const double d{dim_id} = _b{dim_id} > 0.0 ? ({_c_expr(a_expr, sref)}) * _b{dim_id} : 0.0;')
            step_locals.add(dim_id)
        elif op == _OP_LOOKUP:
            if arg in emitted_lookups:
                continue
            emitted_lookups.add(arg)
            lu = spec.rt.lookup_by_id[arg]
            gi = spec.lu_group[arg]
            grp = spec.groups[gi]
            dim_ids = spec.rt._lookup_dim_ids[arg]
            if arg in spec.mem_lus:
                # Group-dispatch fusion + exact structure: the query is the
                # local address for this 4 KiB group; out of range means the
                # empty-group result (0, 0), gate-zeroed downstream.
                m = spec.mem_lus[arg]
                terms = m["terms"]
                qi_src = _c_expr((len(terms), tuple(d for _c, d in terms),
                                  tuple(c for c, _d in terms)), sref)
                emit(f'        const long long qi{arg} = iround({qi_src});')
                emit(f'        double o{arg}_0[2] = {{0.0, 0.0}};')
                emit(f'        if ((unsigned long long)qi{arg} >> 12 == 0) '
                     f'mem_query({m["g"]}, qi{arg}, o{arg}_0);')
                for k, did in enumerate(dim_ids):
                    emit(f'        const double d{did} = o{arg}_0[{k % 2}];')
                    step_locals.add(did)
            elif arg in spec.lu_reg:
                ri = spec.lu_reg[arg]
                for k, did in enumerate(dim_ids):
                    emit(f'        const double d{did} = '
                         f'reg{ri}_any ? reg{ri}_v[{k}] : 0.0;')
                    step_locals.add(did)
            else:
                tb = "TieBreak::LATEST" if grp["latest"] else "TieBreak::AVERAGE"
                qx_expr, qy_expr = spec.rt._lookup_query_exprs[arg]
                emit(f'        const double q{arg}[2] = {{ {_c_expr(qx_expr, sref)}, {_c_expr(qy_expr, sref)} }};')
                for p in range(grp["nvp"]):
                    emit(f'        double o{arg}_{p}[2] = {{0.0, 0.0}};')
                    emit(f'        grp{gi}[{p}].query(q{arg}, {tb}, o{arg}_{p});')
                for k, did in enumerate(dim_ids):
                    emit(f'        const double d{did} = o{arg}_{k // 2}[{k % 2}];')
                    step_locals.add(did)
    for dim_id in spec.rt._external_store_ids:
        emit(f'        v[{dim_id}] = d{dim_id};')
    emit('    }')

    # ── Machine state + MMIO interface ─────────────────────────────
    # The kernel is a persistent machine: load_image() seeds the token log,
    # run() executes until a step budget, halt, jump-to-self trap, or a
    # watched address is touched.  Device reads pause *before* the reading
    # instruction (found by shadow-decoding the RAM mirror) so the harness
    # can inject the register value as a WRITE token; device writes pause
    # after the writing step.  set_nmi() flags the next STATE token; the
    # interrupt sequence itself is graph logic.
    read_mode = ", ".join(str(v) for v in read_mode_table())
    cycles_tab = ", ".join(str(v) for v in cycle_table())
    emit('')
    emit(f'    static constexpr unsigned char READ_MODE[256] = {{ {read_mode} }};')
    emit('    // Base cycles per opcode (no penalty cycles); NMI costs 7.')
    emit(f'    static constexpr unsigned char CYCLES[256] = {{ {cycles_tab} }};')
    emit('')
    emit('    double pos = 0.0;')
    emit('    long step_index = 0;')
    emit('    long cycles_total = 0;')
    emit('    unsigned char ram[65536];  // mirror of the write log, for shadow decode')
    emit('    int st_pc = 0, st_a = 0, st_x = 0, st_y = 0, st_s = 0xFD, st_p = 0x24;')
    emit('    int st_ir = 0, st_oplo = 0, st_ophi = 0, st_ea = 0, st_data = 0;')
    emit('    bool st_halted = false;')
    emit('    bool nmi_pending = false;')
    emit('    bool resume_read_skip = false;')
    emit('    // Read lease: watched reads of this address are served natively')
    emit('    // with this value (device semantics frozen until the bus clears it).')
    emit('    int lease_addr = -1, lease_value = 0;')
    emit('    std::vector<std::pair<int,int>> rwatch, wwatch;')
    emit('')
    emit('    void load_image(py::bytes image_bytes, int start_pc) {')
    emit('        std::string img = image_bytes;')
    emit('        if (img.size() != 65536) throw std::runtime_error("image must be 64 KiB");')
    emit('        reset();')
    emit('        std::memcpy(ram, img.data(), 65536);')
    emit('        pos = 0.0; step_index = 0;')
    emit('        insert_start(start_pc, pos); pos += 1.0;')
    emit('        for (int addr = 0; addr < 65536; ++addr) {')
    emit('            if (ram[addr] == 0) continue;')
    emit('            insert_write(addr, ram[addr], pos); pos += 1.0;')
    emit('        }')
    emit('        st_pc = start_pc & 0xFFFF; st_a = st_x = st_y = 0; st_s = 0xFD; st_p = 0x24;')
    emit('        st_ir = st_oplo = st_ophi = st_ea = st_data = 0;')
    emit('        st_halted = false; nmi_pending = false; resume_read_skip = false;')
    emit('        cycles_total = 0;')
    emit('        rwatch.clear(); wwatch.clear();')
    emit('        lease_addr = -1;')
    emit('    }')
    emit('')
    emit('    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }')
    emit('    void watch_write(int lo, int hi) { wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }')
    emit('    void set_read_lease(int addr, int value) { lease_addr = addr & 0xFFFF; lease_value = value & 0xFF; }')
    emit('    void clear_read_lease() { lease_addr = -1; }')
    emit('    void set_nmi() { nmi_pending = true; }')
    emit('    int read_ram(int addr) { return ram[addr & 0xFFFF]; }')
    emit('    py::bytes dump_ram() { return py::bytes((const char*)ram, 65536); }')
    emit('')
    emit('    // Restore architectural state after a snapshot rebuild (the trace')
    emit('    // fields keep the previous instruction context so a checkpoint is')
    emit('    // invisible even at trace granularity).')
    emit('    void set_state(int pc, int a, int x, int y, int s, int p,')
    emit('                   int ir, int oplo, int ophi, int ea, int data) {')
    emit('        st_pc = pc & 0xFFFF; st_a = a & 0xFF; st_x = x & 0xFF; st_y = y & 0xFF;')
    emit('        st_s = s & 0xFF; st_p = p & 0xFF;')
    emit('        st_ir = ir & 0xFF; st_oplo = oplo & 0xFF; st_ophi = ophi & 0xFF;')
    emit('        st_ea = ea & 0xFFFF; st_data = data & 0xFF;')
    emit('    }')
    emit('')
    emit('    void inject_write(int addr, int value) {')
    emit('        addr &= 0xFFFF; value &= 0xFF;')
    emit('        ram[addr] = (unsigned char)value;')
    emit('        insert_write(addr, value, pos); pos += 1.0;')
    emit('    }')
    emit('')
    emit('    py::dict get_state() {')
    emit('        py::dict st;')
    emit('        st["pc"] = st_pc; st["a"] = st_a; st["x"] = st_x; st["y"] = st_y;')
    emit('        st["s"] = st_s; st["p"] = st_p; st["halted"] = st_halted;')
    emit('        st["ir"] = st_ir; st["oplo"] = st_oplo; st["ophi"] = st_ophi;')
    emit('        st["ea"] = st_ea; st["data"] = st_data;')
    emit('        st["steps"] = step_index; st["cycles"] = cycles_total;')
    emit('        st["pos"] = (long)pos;  // token-log length, drives checkpointing')
    emit('        return st;')
    emit('    }')
    emit('')
    emit('    static bool in_ranges(const std::vector<std::pair<int,int>>& v, int addr) {')
    emit('        for (const auto& r : v) if (addr >= r.first && addr <= r.second) return true;')
    emit('        return false;')
    emit('    }')
    emit('')
    emit('    // Address the instruction at st_pc will read, or -1 (mirrors bus.read_target).')
    emit('    int shadow_read_target() const {')
    emit('        const int mode = READ_MODE[ram[st_pc]];')
    emit('        if (mode == 0) return -1;')
    emit('        const int lo = ram[(st_pc + 1) & 0xFFFF];')
    emit('        const int hi = ram[(st_pc + 2) & 0xFFFF];')
    emit('        switch (mode) {')
    emit('            case 1: return lo;')
    emit('            case 2: return (lo + st_x) & 0xFF;')
    emit('            case 3: return (lo + st_y) & 0xFF;')
    emit('            case 4: return lo | (hi << 8);')
    emit('            case 5: return ((lo | (hi << 8)) + st_x) & 0xFFFF;')
    emit('            case 6: return ((lo | (hi << 8)) + st_y) & 0xFFFF;')
    emit('            case 7: { const int zp = (lo + st_x) & 0xFF;')
    emit('                      return ram[zp] | (ram[(zp + 1) & 0xFF] << 8); }')
    emit('            case 8: { const int base = ram[lo] | (ram[(lo + 1) & 0xFF] << 8);')
    emit('                      return (base + st_y) & 0xFFFF; }')
    emit('        }')
    emit('        return -1;')
    emit('    }')

    # ── Run loop ───────────────────────────────────────────────────
    def out_ref(dim_id):
        return f"v[{dim_id}]" if dim_id in spec.computed else gref(dim_id)

    t = spec.trace_ids
    emit('')
    emit('    py::dict run(long max_steps, bool stop_on_trap, long max_cycles) {')
    emit('        std::vector<int> t_pc, t_a, t_x, t_y, t_s, t_p, t_ir, t_oplo, t_ophi, t_ea, t_data, t_halted;')
    emit('        std::vector<long> w_row; std::vector<int> w_addr, w_value;')
    emit('        std::vector<int> hit_addr, hit_value;')
    emit('        const char* reason = "budget";')
    emit('        int read_addr = -1;')
    emit('        long cycles_this = 0;')
    emit('        double v[NDIMS];')
    emit('        int pend_addr[3], pend_value[3];')
    emit('        for (long i = 0; i < max_steps; ++i) {')
    emit('            if (st_halted) { reason = "halt"; break; }')
    emit('            const bool skip = resume_read_skip;')
    emit('            resume_read_skip = false;')
    emit('            if (!nmi_pending && !skip && !rwatch.empty()) {')
    emit('                const int target = shadow_read_target();')
    emit('                if (target >= 0 && in_ranges(rwatch, target)) {')
    emit('                    if (target != lease_addr) {')
    emit('                        resume_read_skip = true;')
    emit('                        read_addr = target;')
    emit('                        reason = "watch_read";')
    emit('                        break;')
    emit('                    }')
    emit('                    // Leased: serve the device read natively, exactly')
    emit('                    // what the Python pause path would inject.')
    emit('                    inject_write(target, lease_value);')
    emit('                }')
    emit('            }')
    emit('            const bool nmi = nmi_pending;')
    emit('            nmi_pending = false;')
    emit('            insert_state(pos,')
    emit('                (double)(st_pc & 0xFF), (double)((st_pc >> 8) & 0xFF),')
    emit('                (double)st_a, (double)st_x, (double)st_y, (double)st_s,')
    emit('                (double)((st_p >> 0) & 1), (double)((st_p >> 1) & 1), (double)((st_p >> 2) & 1),')
    emit('                (double)((st_p >> 3) & 1), (double)((st_p >> 4) & 1), (double)((st_p >> 5) & 1),')
    emit('                (double)((st_p >> 6) & 1), (double)((st_p >> 7) & 1),')
    emit('                (double)st_ir, (double)st_oplo, (double)st_ophi,')
    emit('                (double)(st_ea & 0xFF), (double)((st_ea >> 8) & 0xFF),')
    emit('                (double)st_data, (double)(st_halted ? 1 : 0), (double)(nmi ? 1 : 0));')
    emit('            pos += 1.0;')
    emit('            step(v);')
    emit(f'            t_pc.push_back((int)(iround(v[{t["current_pc"]}]) & 0xFFFF));')
    emit(f'            t_a.push_back((int)(iround(v[{t["new_a"]}]) & 0xFF));')
    emit(f'            t_x.push_back((int)(iround(v[{t["new_x"]}]) & 0xFF));')
    emit(f'            t_y.push_back((int)(iround(v[{t["new_y"]}]) & 0xFF));')
    emit(f'            t_s.push_back((int)(iround(v[{t["new_s"]}]) & 0xFF));')
    emit(f'            t_p.push_back((int)(iround(v[{t["new_p"]}]) & 0xFF));')
    emit(f'            t_ir.push_back((int)(iround(v[{t["fetched_opcode"]}]) & 0xFF));')
    emit(f'            t_oplo.push_back((int)(iround(v[{t["fetched_operand_lo"]}]) & 0xFF));')
    emit(f'            t_ophi.push_back((int)(iround(v[{t["fetched_operand_hi"]}]) & 0xFF));')
    emit(f'            t_ea.push_back((int)(iround(v[{t["ea"]}]) & 0xFFFF));')
    emit(f'            t_data.push_back((int)(iround(v[{t["data_out"]}]) & 0xFF));')
    emit(f'            const bool halted = iround(v[{t["new_halted"]}]) != 0;')
    emit('            t_halted.push_back(halted ? 1 : 0);')
    emit('            int n_writes = 0, n_hits = 0;')
    emit('            int sh_addr[3], sh_value[3];')
    for store, addr, value in (
        ("is_store", "write_addr_out", "write_value_out"),
        ("is_store2", "write_addr2_out", "write_value2_out"),
        ("is_store3", "write_addr3_out", "write_value3_out"),
    ):
        emit(f'            if (iround(v[{t[store]}]) != 0) {{')
        emit(f'                pend_addr[n_writes] = (int)(iround(v[{t[addr]}]) & 0xFFFF);')
        emit(f'                pend_value[n_writes] = (int)(iround(v[{t[value]}]) & 0xFF);')
        emit('                w_row.push_back(step_index); w_addr.push_back(pend_addr[n_writes]); w_value.push_back(pend_value[n_writes]);')
        emit('                ++n_writes;')
        emit('            }')
    emit('            for (int w = 0; w < n_writes; ++w) {')
    emit('                ram[pend_addr[w]] = (unsigned char)pend_value[w];')
    emit('                insert_write(pend_addr[w], pend_value[w], pos); pos += 1.0;')
    emit('                if (!wwatch.empty() && in_ranges(wwatch, pend_addr[w])) {')
    emit('                    sh_addr[n_hits] = pend_addr[w]; sh_value[n_hits] = pend_value[w]; ++n_hits;')
    emit('                }')
    emit('            }')
    emit(f'            const long next_pc = iround(v[{t["next_pc"]}]) & 0xFFFF;')
    emit(f'            const long p = iround(v[{t["new_p"]}]) & 0xFF;')
    emit('            const int prev_pc = st_pc;')
    emit('            st_pc = (int)next_pc;')
    emit('            st_a = t_a.back(); st_x = t_x.back(); st_y = t_y.back(); st_s = t_s.back();')
    emit('            st_p = (int)p;')
    emit('            st_ir = t_ir.back(); st_oplo = t_oplo.back(); st_ophi = t_ophi.back();')
    emit('            st_ea = t_ea.back(); st_data = t_data.back();')
    emit('            ++step_index;')
    emit('            const long step_cycles = nmi ? 7 : (long)CYCLES[(unsigned char)t_ir.back()];')
    emit('            cycles_this += step_cycles; cycles_total += step_cycles;')
    emit('            for (int k = 0; k < n_hits; ++k) { hit_addr.push_back(sh_addr[k]); hit_value.push_back(sh_value[k]); }')
    emit('            if (halted) { st_halted = true; reason = "halt"; break; }')
    score_names = []
    for name, expr in spec.outputs:
        var = f"score_{name.lower()}"
        score_names.append((name, var))
        emit(f'            const double {var} = {_c_expr(expr, out_ref)};')
    if [n for n, _ in spec.outputs] != ["STATE", "HALT"]:
        raise NotImplementedError(f"unexpected output tokens: {[n for n, _ in spec.outputs]}")
    emit(f'            if ({score_names[1][1]} > {score_names[0][1]}) {{ st_halted = true; reason = "halt"; break; }}')
    emit('            if (stop_on_trap && next_pc == (long)prev_pc) { reason = "trap"; break; }')
    emit('            if (n_hits > 0) { reason = "watch_write"; break; }')
    emit('            if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }')
    emit('        }')
    emit('')
    emit('        py::dict out;')
    emit('        auto arr = [](const std::vector<int>& src) {')
    emit('            return py::array_t<int>((py::ssize_t)src.size(), src.data());')
    emit('        };')
    for field in ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi", "ea", "data", "halted"):
        emit(f'        out["{field}"] = arr(t_{field});')
    emit('        out["w_row"] = py::array_t<long>((py::ssize_t)w_row.size(), w_row.data());')
    emit('        out["w_addr"] = arr(w_addr);')
    emit('        out["w_value"] = arr(w_value);')
    emit('        out["hit_addr"] = arr(hit_addr);')
    emit('        out["hit_value"] = arr(hit_value);')
    emit('        out["read_addr"] = read_addr;')
    emit('        out["reason"] = reason;')
    emit('        out["cycles"] = cycles_this;')
    emit('        return out;')
    emit('    }')
    emit('};')
    emit('')
    emit('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {')
    emit('    py::class_<KERNEL_T>(m, "Kernel", py::module_local())')
    emit('        .def(py::init<>())')
    emit('        .def("load_image", &KERNEL_T::load_image, py::arg("image"), py::arg("start_pc"))')
    emit('        .def("run", &KERNEL_T::run, py::arg("max_steps"), py::arg("stop_on_trap") = false,')
    emit('             py::arg("max_cycles") = 0L)')
    emit('        .def("watch_read", &KERNEL_T::watch_read)')
    emit('        .def("watch_write", &KERNEL_T::watch_write)')
    emit('        .def("set_read_lease", &KERNEL_T::set_read_lease)')
    emit('        .def("clear_read_lease", &KERNEL_T::clear_read_lease)')
    emit('        .def("set_nmi", &KERNEL_T::set_nmi)')
    emit('        .def("inject_write", &KERNEL_T::inject_write)')
    emit('        .def("read_ram", &KERNEL_T::read_ram)')
    emit('        .def("dump_ram", &KERNEL_T::dump_ram)')
    emit('        .def("set_state", &KERNEL_T::set_state)')
    emit('        .def("get_state", &KERNEL_T::get_state);')
    emit('}')
    emit('')
    return "\n".join(L)


class Cpu6502CppRuntime:
    """Graph evaluator running entirely in generated C++."""

    def __init__(self, *, bcd: bool = True):
        rt = Cpu6502Runtime(bcd=bcd)
        source = generate_cpp(rt)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        source = source.replace("KERNEL_T", f"Kernel_{digest}")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        src_path = os.path.join(_CACHE_DIR, f"kernel_{digest}.cpp")
        if not os.path.exists(src_path):
            with open(src_path, "w") as f:
                f.write(source)

        from torch.utils.cpp_extension import load

        build_start = time.perf_counter()
        module = load(
            name=f"cpu6502_kernel_{digest}",
            sources=[src_path],
            extra_cflags=["-O2", "-std=c++17"],
            extra_include_paths=[_ATTENTION_DIR],
            verbose=False,
        )
        self.compile_seconds = time.perf_counter() - build_start
        self._kernel = module.Kernel()
        self.last_prefix_seconds = 0.0
        self.last_step_seconds = 0.0

    # ── Machine interface (mirrors evaluator.py's; drives the C++ side) ──

    def load_image(self, memory_image: bytes | bytearray, *, start_pc: int = 0x0400):
        if len(memory_image) != 0x10000:
            raise ValueError("memory_image must be exactly 64 KiB")
        prefix_start = time.perf_counter()
        self._kernel.load_image(bytes(memory_image), start_pc)
        self.last_prefix_seconds = time.perf_counter() - prefix_start

    def run(self, max_steps: int, *, stop_on_trap: bool = False,
            max_cycles: int = 0) -> dict:
        """Run until a budget, halt, trap, or a watched address is touched.

        max_cycles > 0 additionally stops once that many base CPU cycles
        have elapsed in this call (reason "cycles"; used for frame pacing).
        Returns trace arrays plus "reason" ("budget" | "halt" | "trap" |
        "watch_read" | "watch_write" | "cycles"), "cycles" (elapsed this
        call), "read_addr" (pending device read, or -1), and
        "hit_addr"/"hit_value" (watched writes from the final step).
        """
        step_start = time.perf_counter()
        out = self._kernel.run(max_steps, stop_on_trap, max_cycles)
        self.last_step_seconds += time.perf_counter() - step_start
        return out

    def watch_read(self, lo: int, hi: int):
        self._kernel.watch_read(lo, hi)

    def watch_write(self, lo: int, hi: int):
        self._kernel.watch_write(lo, hi)

    def set_read_lease(self, addr: int, value: int):
        self._kernel.set_read_lease(addr, value)

    def clear_read_lease(self):
        self._kernel.clear_read_lease()

    def set_nmi(self):
        self._kernel.set_nmi()

    def inject_write(self, addr: int, value: int):
        self._kernel.inject_write(addr, value)

    def read_ram(self, addr: int) -> int:
        return self._kernel.read_ram(addr)

    def dump_ram(self) -> bytes:
        return self._kernel.dump_ram()

    def set_state(self, state: dict) -> None:
        self._kernel.set_state(
            state["pc"], state["a"], state["x"], state["y"], state["s"], state["p"],
            state.get("ir", 0), state.get("oplo", 0), state.get("ophi", 0),
            state.get("ea", 0), state.get("data", 0),
        )

    def get_state(self) -> dict:
        return self._kernel.get_state()

    # ── One-shot runs (Klaus / nestest harnesses) ───────────────────

    def run_image_arrays(self, memory_image: bytes | bytearray, *,
                         start_pc: int = 0x0400, max_steps: int = 100_000,
                         stop_on_trap: bool = False) -> dict:
        self.load_image(memory_image, start_pc=start_pc)
        self.last_step_seconds = 0.0
        return self.run(max_steps, stop_on_trap=stop_on_trap)

    def run_image(self, memory_image: bytes | bytearray, *,
                  start_pc: int = 0x0400, max_steps: int = 100_000) -> Trace:
        out = self.run_image_arrays(memory_image, start_pc=start_pc, max_steps=max_steps)
        return trace_from_arrays(out)


def trace_from_arrays(out: dict) -> Trace:
    """Convert kernel result arrays into the canonical Trace format."""
    writes_by_row: dict[int, list[MemWrite]] = {}
    for row, addr, value in zip(out["w_row"], out["w_addr"], out["w_value"], strict=True):
        writes_by_row.setdefault(int(row), []).append(MemWrite(int(row), int(addr), int(value)))
    trace = Trace()
    for i in range(len(out["pc"])):
        trace.append(TraceRow(
            step=i,
            state=Cpu6502State(
                pc=int(out["pc"][i]), a=int(out["a"][i]), x=int(out["x"][i]),
                y=int(out["y"][i]), s=int(out["s"][i]), p=int(out["p"][i]),
                ir=int(out["ir"][i]), operand_lo=int(out["oplo"][i]),
                operand_hi=int(out["ophi"][i]), ea=int(out["ea"][i]),
                data=int(out["data"][i]), halted=bool(out["halted"][i]),
            ),
            writes=tuple(writes_by_row.get(i, ())),
        ))
    return trace
