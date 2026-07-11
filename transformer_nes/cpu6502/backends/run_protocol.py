"""Shared tables and C++ blocks for the native CPU run loops.

kernel.py (graph kernel) and model/engine_gen.py (weights engine) emit the
same persistent-machine protocol: one exact memory structure, one shadow
decoder, one run loop (watch/lease/trace/budget), one machine API.  The
KernelBus contract is the spec; emitting these blocks from one place keeps
the two engines from drifting.
"""

from __future__ import annotations

from transformer_nes.cpu6502.spec import (
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

READ_MODE_CODE = {
    MODE_ZERO_PAGE: 1,
    MODE_ZERO_PAGE_X: 2,
    MODE_ZERO_PAGE_Y: 3,
    MODE_ABSOLUTE: 4,
    MODE_ABSOLUTE_X: 5,
    MODE_ABSOLUTE_Y: 6,
    MODE_INDIRECT_X: 7,
    MODE_INDIRECT_Y: 8,
}


def read_mode_table() -> tuple[int, ...]:
    """Opcode-to-addressing-mode codes used by MMIO shadow decoding."""
    table = [0] * 256
    for byte, op in OPCODES.items():
        if op.operation in READS_MEMORY:
            table[byte] = READ_MODE_CODE.get(op.mode, 0)
    return tuple(table)


def cycle_table() -> tuple[int, ...]:
    """Base opcode cycles; dynamic penalties are outside the run protocol."""
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = op.cycles
    return tuple(table)


MEM_QUERY_TEMPLATE = """\
    // Exact memory-head argmax: nearest address wins; equal-distance
    // ties use the latest write. One structure serves each 4 KiB group,
    // so attention state is O(1) in the token log.
    int mem_val[16][4096];
    long long mem_seq[16][4096];
    unsigned long long mem_bits[16][64];

    void mem_query(const int g_, const long long q, {OUT} out[2]) const {
        const unsigned long long* bits = mem_bits[g_];
        // Written addresses answer for themselves: below == above == q,
        // and the tie-break picks the same entry either way.
        if (q >= 0 && q <= 4095 && (bits[q >> 6] >> (q & 63)) & 1ULL) {
            out[0] = {CAST}mem_val[g_][q];
            out[1] = {CAST}(((long long)g_ << 12) + q);
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
        if (below < 0 && above < 0) { out[0] = {ZERO}; out[1] = {ZERO}; return; }
        else if (below < 0) win = above;
        else if (above < 0) win = below;
        else {
            const long long db = q - below, da = above - q;
            if (db < da) win = below;
            else if (da < db) win = above;
            else win = mem_seq[g_][below] > mem_seq[g_][above] ? below : above;
        }
        out[0] = {CAST}mem_val[g_][win];
        out[1] = {CAST}(((long long)g_ << 12) + win);
    }"""


def mem_query_cpp(out_type: str) -> str:
    """The exact nearest-written-address structure (members + query);
    ``out_type`` is the caller's output element type."""
    cast = "" if out_type == "long long" else f"({out_type})"
    zero = "0.0" if out_type == "double" else "0"
    return (
        MEM_QUERY_TEMPLATE
        .replace("{OUT}", out_type)
        .replace("{CAST}", cast)
        .replace("{ZERO}", zero)
    )


SHADOW_READ_TARGET_CPP = """\
    // Shadow-decode the instruction at st_pc; return the address it
    // will read, or -1 (mirrors nes/bus read_target).
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
    }"""

IN_RANGES_CPP = """\
    static bool in_ranges(const std::vector<std::pair<int,int>>& v, int addr) {
        for (const auto& r : v) if (addr >= r.first && addr <= r.second) return true;
        return false;
    }"""


def run_state_members_cpp() -> str:
    """Persistent-machine state shared by both run loops: RAM mirror,
    architectural registers, pause/lease flags, counters, watch ranges, and
    the shadow-decode / base-cycle tables."""
    read_mode = ", ".join(str(v) for v in read_mode_table())
    cycles_tab = ", ".join(str(v) for v in cycle_table())
    return f"""\
    unsigned char ram[65536];  // mirror of the write log, for shadow decode
    int st_pc = 0, st_a = 0, st_x = 0, st_y = 0, st_s = 0xFD, st_p = 0x24;
    int st_ir = 0, st_oplo = 0, st_ophi = 0, st_ea = 0, st_data = 0;
    bool st_halted = false;
    bool nmi_pending = false;
    bool resume_read_skip = false;
    // Read lease: watched reads of this address are served natively
    // with this value (device semantics frozen until the bus clears it).
    int lease_addr = -1, lease_value = 0;
    long step_index = 0;
    long cycles_total = 0;
    std::vector<std::pair<int,int>> rwatch, wwatch;
    static constexpr unsigned char READ_MODE[256] = {{ {read_mode} }};
    // Base cycles per opcode (no penalty cycles); NMI costs 7.
    static constexpr unsigned char CYCLES[256] = {{ {cycles_tab} }};"""


MACHINE_API_CPP = """\
    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void watch_write(int lo, int hi) { wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void clear_watches() { rwatch.clear(); wwatch.clear(); }
    void set_read_lease(int addr, int value) { lease_addr = addr & 0xFFFF; lease_value = value & 0xFF; }
    void clear_read_lease() { lease_addr = -1; }
    void set_nmi() { nmi_pending = true; }
    int read_ram(int addr) { return ram[addr & 0xFFFF]; }
    py::bytes dump_ram() { return py::bytes((const char*)ram, 65536); }"""


RUN_LOOP_TEMPLATE = """\
    // Run up to max_steps instructions natively, pausing back to
    // Python only on halt / jump-to-self trap / watched device
    // access / cycle budget.  One protocol and result dict for both
    // generated engines; the KernelBus contract is the spec.
    py::dict run(long max_steps, bool stop_on_trap, long max_cycles) {
        std::vector<int> t_pc, t_a, t_x, t_y, t_s, t_p, t_ir, t_oplo, t_ophi, t_ea, t_data, t_halted;
        std::vector<long> w_row; std::vector<int> w_addr, w_value;
        std::vector<int> hit_addr, hit_value;
        const char* reason = "budget";
        int read_addr = -1;
        long cycles_this = 0;
{EXTRA_LOCALS}
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
{STEP_BODY}
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
    }"""


def run_loop_cpp(*, extra_locals: str, step_body: str) -> str:
    """The shared run loop; the engine supplies only its locals and the
    per-iteration step body (everything between the watch/lease check and
    the loop's closing brace)."""
    return (
        RUN_LOOP_TEMPLATE
        .replace("{EXTRA_LOCALS}", extra_locals)
        .replace("{STEP_BODY}", step_body)
    )
