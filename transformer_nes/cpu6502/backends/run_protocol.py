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
    """Base opcode cycles; dynamic penalties are outside the run protocol.

    Unknown bytes cost 2, never 0: a zero-cycle step can never satisfy a
    cycle budget, so garbage execution (a bad pointer landing between
    opcodes) would spin the native run loop forever instead of returning
    at the frame boundary."""
    table = [2] * 256
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
    // IRQ line: delivered at a step boundary only while the I flag is
    // clear (stays pending until then); NMI wins when both are set.
    bool irq_pending = false;
    bool resume_read_skip = false;
    // Read leases: watched reads of these addresses are served natively.
    // n == 0 repeats one value (device semantics frozen until the bus
    // clears it); otherwise a finite value queue (controller serial
    // reads) whose exhaustion pauses, with served reads counted so the
    // bus can advance the device model.  Multiple slots keep interleaved
    // port reads ($4016/$4017 bit pairs) from thrashing one slot.
    static constexpr int NLEASE = 4;
    int lease_addr[NLEASE] = {{-1, -1, -1, -1}};
    int lease_value[NLEASE] = {{0, 0, 0, 0}};
    int lease_vals[NLEASE][8];
    int lease_n[NLEASE] = {{0, 0, 0, 0}};
    int lease_i[NLEASE] = {{0, 0, 0, 0}};
    long lease_served[NLEASE] = {{0, 0, 0, 0}};
    long step_index = 0;
    long cycles_total = 0;
    std::vector<std::pair<int,int>> rwatch, wwatch;
    // Lowest watched-write address (0x10000 = none): store-licensed
    // windows dispatch only while this stays >= 0x2000, making a watched
    // write inside a window statically impossible.
    int watch_write_lo = 0x10000;
    // Native cartridge PRG banking (console silicon, like the PPU):
    // $8000-$FFFF register writes decode natively -- bank switches copy
    // from the resident PRG ROM, the clobbered ROM byte is repaired --
    // and are journaled for the Python mapper's CHR/IRQ/mirroring side
    // effects, drained at every run() return.  The frame-at-once PPU
    // samples that state only at split/render boundaries, which all
    // follow a drain, so the deferral is invisible.
    int mapper_kind = 0;  // 0 none, 1 MMC1, 3 MMC3
    std::vector<unsigned char> mapper_prg;
    int mapper_nbanks = 0;
    int mapper_cur[4] = {{-1, -1, -1, -1}};  // 8K slot banks (MMC1: 16K pairs)
    int m3_bank_select = 0;
    int m3_regs[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};
    int m1_shift = 0x10, m1_control = 0x0C, m1_prg = 0;
    std::vector<int> mj_addr, mj_value;  // journal since last drain
    static constexpr unsigned char READ_MODE[256] = {{ {read_mode} }};
    // Base cycles per opcode (no penalty cycles); NMI costs 7.
    static constexpr unsigned char CYCLES[256] = {{ {cycles_tab} }};"""


MACHINE_API_CPP = """\
    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void watch_write(int lo, int hi) {
        wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF);
        if ((lo & 0xFFFF) < watch_write_lo) watch_write_lo = lo & 0xFFFF;
    }
    void clear_watches() { rwatch.clear(); wwatch.clear(); watch_write_lo = 0x10000; }
    int lease_slot(int addr) const {
        for (int s = 0; s < NLEASE; ++s) if (lease_addr[s] == addr) return s;
        return -1;
    }
    // Reuse the address's slot, else a free one, else an exhausted queue,
    // else slot 0.  The bus flushes served counts before every set, so an
    // eviction never loses a device advance.
    int lease_claim(int addr) {
        int s = lease_slot(addr);
        if (s < 0) s = lease_slot(-1);
        if (s < 0)
            for (int t = 0; t < NLEASE; ++t)
                if (lease_n[t] > 0 && lease_i[t] == lease_n[t]) { s = t; break; }
        return s < 0 ? 0 : s;
    }
    // Both setters return the evicted address (or -1) so the bus can drop
    // its device tracking for it.
    int set_read_lease(int addr, int value) {
        addr &= 0xFFFF;
        const int s = lease_claim(addr);
        const int evicted = lease_addr[s] == addr ? -1 : lease_addr[s];
        lease_addr[s] = addr; lease_value[s] = value & 0xFF;
        lease_n[s] = lease_i[s] = 0; lease_served[s] = 0;
        return evicted;
    }
    int set_read_lease_seq(int addr, py::bytes values) {
        const std::string v = values;
        addr &= 0xFFFF;
        const int s = lease_claim(addr);
        const int evicted = lease_addr[s] == addr ? -1 : lease_addr[s];
        lease_addr[s] = addr;
        lease_n[s] = v.size() < 8 ? (int)v.size() : 8;
        lease_i[s] = 0; lease_served[s] = 0;
        for (int i = 0; i < lease_n[s]; ++i) lease_vals[s][i] = (unsigned char)v[i];
        if (lease_n[s] == 0) lease_addr[s] = -1;
        return evicted;
    }
    void clear_read_lease() {
        for (int s = 0; s < NLEASE; ++s) {
            lease_addr[s] = -1; lease_n[s] = lease_i[s] = 0; lease_served[s] = 0;
        }
    }
    // Targeted invalidation (a write only disturbs its own device's
    // registers); the bus flushes served counts first.
    void clear_read_lease_addr(int addr) {
        const int s = lease_slot(addr & 0xFFFF);
        if (s >= 0) {
            lease_addr[s] = -1; lease_n[s] = lease_i[s] = 0; lease_served[s] = 0;
        }
    }
    long take_lease_served(int addr) {
        const int s = lease_slot(addr & 0xFFFF);
        if (s < 0) return 0;
        const long n = lease_served[s]; lease_served[s] = 0; return n;
    }
    void set_nmi() { nmi_pending = true; }
    void set_irq() { irq_pending = true; }
    void clear_irq() { irq_pending = false; }
    // IRQ deliverable now: I clear and no NMI ahead of it.
    bool irq_ready() const { return irq_pending && !nmi_pending && !(st_p & 0x04); }
    int read_ram(int addr) { return ram[addr & 0xFFFF]; }
    py::bytes dump_ram() { return py::bytes((const char*)ram, 65536); }

    // ── Native PRG banking ───────────────────────────────────────────
    void mapper_slot(int slot, int bank, int size) {
        if (mapper_cur[slot] == bank) return;
        mapper_cur[slot] = bank;
        const unsigned char* src = mapper_prg.data() + (size_t)bank * size;
        const int base = 0x8000 + slot * size;
        for (int i = 0; i < size; ++i) map_write(base + i, src[i]);
    }
    void mapper_apply_mmc3() {
        const int n = mapper_nbanks;
        const int r6 = m3_regs[6] % n, r7 = m3_regs[7] % n;
        const int second_last = n - 2, last = n - 1;
        const bool swap = m3_bank_select & 0x40;
        mapper_slot(0, swap ? second_last : r6, 0x2000);
        mapper_slot(1, r7, 0x2000);
        mapper_slot(2, swap ? r6 : second_last, 0x2000);
        mapper_slot(3, last, 0x2000);
    }
    void mapper_apply_mmc1() {
        const int n = mapper_nbanks;
        const int mode = (m1_control >> 2) & 3;
        const int bank = (m1_prg & 0x0F) % n;
        int lo, hi;
        if (mode <= 1) { lo = bank & 0x0E; hi = (bank & 0x0E) | 1; }
        else if (mode == 2) { lo = 0; hi = bank; }
        else { lo = bank; hi = n - 1; }
        mapper_slot(0, lo, 0x4000);
        mapper_slot(1, hi, 0x4000);
    }
    int mapper_rom_byte(int addr) {
        if (mapper_kind == 1) {
            const int bank = mapper_cur[(addr >> 14) & 1];
            return mapper_prg[(size_t)bank * 0x4000 + (addr & 0x3FFF)];
        }
        const int bank = mapper_cur[(addr - 0x8000) >> 13];
        return mapper_prg[(size_t)bank * 0x2000 + (addr & 0x1FFF)];
    }
    // Decode a $8000-$FFFF register write, apply any PRG switch, repair
    // the clobbered ROM byte, journal for the Python mapper's CHR/IRQ/
    // mirroring side effects.
    void mapper_write(int addr, int value) {
        mj_addr.push_back(addr); mj_value.push_back(value);
        if (mapper_kind == 3) {
            const int reg = addr & 0xE001;
            if (reg == 0x8000) { m3_bank_select = value; mapper_apply_mmc3(); }
            else if (reg == 0x8001) {
                m3_regs[m3_bank_select & 7] = value & 0xFF;
                mapper_apply_mmc3();
            } else if (reg == 0xE000) {
                irq_pending = false;  // $E000 acks the line NOW; the
            }                         // drain reconciles console sources
        } else if (value & 0x80) {
            m1_shift = 0x10; m1_control |= 0x0C;
            mapper_apply_mmc1();
        } else {
            const bool complete = m1_shift & 1;
            m1_shift = ((m1_shift >> 1) | ((value & 1) << 4)) & 0x1F;
            if (complete) {
                const int reg = (addr >> 13) & 3;
                if (reg == 0) m1_control = m1_shift;
                else if (reg == 3) m1_prg = m1_shift;
                m1_shift = 0x10;
                mapper_apply_mmc1();
            }
        }
        map_write(addr, mapper_rom_byte(addr));
    }
    void set_mapper_mmc3(py::bytes prg, int bank_select, py::bytes regs,
                         py::bytes cur) {
        const std::string p = prg, r = regs, c = cur;
        mapper_prg.assign(p.begin(), p.end());
        mapper_nbanks = (int)(p.size() / 0x2000);
        m3_bank_select = bank_select;
        for (int i = 0; i < 8 && i < (int)r.size(); ++i)
            m3_regs[i] = (unsigned char)r[i];
        for (int i = 0; i < 4 && i < (int)c.size(); ++i)
            mapper_cur[i] = (unsigned char)c[i];
        mapper_kind = 3;
    }
    void set_mapper_mmc1(py::bytes prg, int shift, int control, int prgreg,
                         int lo, int hi) {
        const std::string p = prg;
        mapper_prg.assign(p.begin(), p.end());
        mapper_nbanks = (int)(p.size() / 0x4000);
        m1_shift = shift; m1_control = control; m1_prg = prgreg;
        mapper_cur[0] = lo; mapper_cur[1] = hi;
        mapper_kind = 1;
    }
    void clear_mapper() {
        mapper_kind = 0; mapper_prg.clear();
        for (int i = 0; i < 4; ++i) mapper_cur[i] = -1;
        m1_shift = 0x10; m1_control = 0x0C; m1_prg = 0;
        m3_bank_select = 0;
        for (int i = 0; i < 8; ++i) m3_regs[i] = 0;
        mj_addr.clear(); mj_value.clear();
    }"""


RUN_LOOP_TEMPLATE = """\
    // Run up to max_steps instructions natively, pausing back to
    // Python only on halt / jump-to-self trap / watched device
    // access / cycle budget.  One protocol and result dict for both
    // generated engines; the KernelBus contract is the spec.
    // trace=false skips the per-step trace/write arrays (the console's
    // gameplay path reads only steps/hits/reason/cycles).
    py::dict run(long max_steps, bool stop_on_trap, long max_cycles, bool trace) {
        std::vector<int> t_pc, t_a, t_x, t_y, t_s, t_p, t_ir, t_oplo, t_ophi, t_ea, t_data, t_halted;
        std::vector<long> w_row; std::vector<int> w_addr, w_value;
        std::vector<int> hit_addr, hit_value;
        const char* reason = "budget";
        int read_addr = -1;
        long cycles_this = 0;
        const long steps_entry = step_index;
{EXTRA_LOCALS}
        for (long i = 0; i < max_steps; ++i) {
            if (st_halted) { reason = "halt"; break; }
            const bool skip = resume_read_skip;
            resume_read_skip = false;
            if (!nmi_pending && !irq_ready() && !skip && !rwatch.empty()) {
                const int target = shadow_read_target();
                if (target >= 0 && in_ranges(rwatch, target)) {
                    // Leased: serve the device read natively, exactly
                    // what the Python pause path would inject.  An
                    // exhausted queue keeps its slot (preserving the
                    // served count for the bus) but pauses.
                    bool served = false;
                    const int ls = lease_slot(target);
                    if (ls >= 0) {
                        if (lease_n[ls] == 0) {
                            inject_write(target, lease_value[ls]); served = true;
                        } else if (lease_i[ls] < lease_n[ls]) {
                            inject_write(target, lease_vals[ls][lease_i[ls]++]);
                            ++lease_served[ls]; served = true;
                        }
                    }
                    if (!served) {
                        resume_read_skip = true; read_addr = target; reason = "watch_read"; break;
                    }
                }
            }
{STEP_BODY}
        }
        py::dict out;
        auto arr = [](const std::vector<int>& src) {
            return py::array_t<int>((py::ssize_t)src.size(), src.data());
        };
        // Untraced runs (the console's gameplay path pauses 100+ times a
        // frame) return only what the bus reads; the 12 empty per-step
        // arrays measurably cost at that pause rate.
        if (trace) {
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
        }
        out["hit_addr"] = arr(hit_addr);
        out["hit_value"] = arr(hit_value);
        // Natively applied mapper writes since the last return: the bus
        // drains them into the Python mapper's side-effect path.
        out["m_addr"] = arr(mj_addr);
        out["m_value"] = arr(mj_value);
        mj_addr.clear(); mj_value.clear();
        out["read_addr"] = read_addr;
        out["reason"] = reason;
        out["cycles"] = cycles_this;
        out["steps"] = step_index - steps_entry;
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
