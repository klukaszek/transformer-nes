"""Tables and C++ blocks for the weights engine's native run loop.

model/engine_gen.py emits this persistent-machine protocol into the
generated engine: one exact memory structure, one shadow decoder, one run
loop (watch/lease/trace/budget), one machine API and its native ABI.
The consumer is the console's EngineMachine (nes/bindings.cpp); keeping
the blocks here keeps the protocol reviewable in one place.
"""

from __future__ import annotations

from transformer_nes.cpu6502.backends.weights.windows import WINDOW_EXCLUDED, ends_window
from transformer_nes.cpu6502.spec import (
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_ACCUMULATOR,
    MODE_IMMEDIATE,
    MODE_INDIRECT,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_RELATIVE,
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


TRACE_MODE_CODE = {
    MODE_IMMEDIATE: 11,
    MODE_ZERO_PAGE: 1,
    MODE_ZERO_PAGE_X: 2,
    MODE_ZERO_PAGE_Y: 3,
    MODE_ABSOLUTE: 4,
    MODE_ABSOLUTE_X: 5,
    MODE_ABSOLUTE_Y: 6,
    MODE_INDIRECT_X: 7,
    MODE_INDIRECT_Y: 8,
    MODE_INDIRECT: 9,
    MODE_RELATIVE: 10,
}


def trace_mode_table() -> tuple[int, ...]:
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = TRACE_MODE_CODE.get(op.mode, 0)
    return tuple(table)


def trace_length_table() -> tuple[int, ...]:
    table = [1] * 256
    for byte, op in OPCODES.items():
        table[byte] = op.length
    return tuple(table)


def ends_window_table() -> tuple[int, ...]:
    """Opcodes after which the next instruction starts a new block: what
    ends a learned window (branches, returns), what none holds
    (windows.py WINDOW_EXCLUDED, indirect JMP), and undefined bytes -- 2
    for a conditional branch, whose outcomes the profile also counts."""
    table = [1] * 256
    for byte, op in OPCODES.items():
        table[byte] = 2 if op.mode == "rel" else int(
            ends_window(op) or op.operation in WINDOW_EXCLUDED
            or (op.operation == "jmp" and op.mode != "abs")
        )
    return tuple(table)


def trace_data_table() -> tuple[int, ...]:
    """Opcode-to-source code for reference-compatible trace ``data``."""
    pre_a = {"store_a", "pha"}
    pre_x = {"store_x"}
    pre_y = {"store_y"}
    pre_p = {"php"}
    write = {
        "inc", "isb", "dec", "dcp", "sax", "slo", "rla", "sre", "rra",
    }
    post_a = {
        "load_a", "lax", "adc", "sbc", "and", "ora", "eor", "pla",
        "txa", "tya",
    }
    post_x = {"load_x", "inx", "dex", "tax", "tsx"}
    post_y = {"load_y", "iny", "dey", "tay"}
    table = [0] * 256
    for byte, op in OPCODES.items():
        operation = op.operation
        if operation in pre_a:
            kind = 1
        elif operation in pre_x:
            kind = 2
        elif operation in pre_y:
            kind = 3
        elif operation in pre_p:
            kind = 4
        elif operation in write:
            kind = 5
        elif operation in {"asl", "lsr", "rol", "ror"}:
            kind = 6 if op.mode == MODE_ACCUMULATOR else 5
        elif operation in post_a:
            kind = 6
        elif operation in post_x:
            kind = 7
        elif operation in post_y:
            kind = 8
        elif operation == "txs":
            kind = 9
        elif operation == "plp":
            kind = 10
        elif operation == "cmp":
            kind = 11
        elif operation == "cpx":
            kind = 12
        elif operation == "cpy":
            kind = 13
        elif operation == "bit":
            kind = 14
        else:
            kind = 0
        table[byte] = kind
    return tuple(table)


MEM_QUERY_TEMPLATE = """\
    // Exact memory-head argmax: nearest address wins; equal-distance
    // ties use the latest write. One structure serves each 4 KiB group,
    // so attention state is O(1) in the token log.
    int mem_val[{GROUPS}][4096];
    long long mem_seq[{GROUPS}][4096];
    unsigned long long mem_bits[{GROUPS}][64];

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


def mem_query_cpp(out_type: str, groups: int = 16) -> str:
    """The exact nearest-written-address structure (members + query) over
    ``groups`` 4 KiB groups; ``out_type`` is the caller's output element
    type."""
    cast = "" if out_type == "long long" else f"({out_type})"
    zero = "0.0" if out_type == "double" else "0"
    return (
        MEM_QUERY_TEMPLATE
        .replace("{OUT}", out_type)
        .replace("{CAST}", cast)
        .replace("{ZERO}", zero)
        .replace("{GROUPS}", str(groups))
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


TRACE_RECEIPT_CPP = """\
    int trace_ea(int opcode, int pc, int xr, int yr) const {
        const int mode = TRACE_MODE[opcode];
        const int lo = ram[(pc + 1) & 0xFFFF];
        const int hi = ram[(pc + 2) & 0xFFFF];
        switch (mode) {
            case 1: case 11: return lo;
            case 2: return (lo + xr) & 0xFF;
            case 3: return (lo + yr) & 0xFF;
            case 4: return lo | (hi << 8);
            case 5: return ((lo | (hi << 8)) + xr) & 0xFFFF;
            case 6: return ((lo | (hi << 8)) + yr) & 0xFFFF;
            case 7: { const int zp = (lo + xr) & 0xFF;
                      return ram[zp] | (ram[(zp + 1) & 0xFF] << 8); }
            case 8: { const int base = ram[lo] | (ram[(lo + 1) & 0xFF] << 8);
                      return (base + yr) & 0xFFFF; }
            case 9: { const int ptr = lo | (hi << 8);
                      const int next = (ptr & 0xFF00) | ((ptr + 1) & 0xFF);
                      return ram[ptr] | (ram[next] << 8); }
            case 10: return (pc + 2 + (int)(std::int8_t)lo) & 0xFFFF;
        }
        return 0;
    }

    int trace_data(int opcode, int pre_a, int pre_x, int pre_y, int pre_p,
                   int operand) const {
        switch (TRACE_DATA[opcode]) {
            case 1: return pre_a;
            case 2: return pre_x;
            case 3: return pre_y;
            case 4: return pre_p;
            case 5: return exec_write_value[0];
            case 6: return st_a;
            case 7: return st_x;
            case 8: return st_y;
            case 9: return st_s;
            case 10: return st_p;
            case 11: return (pre_a - operand) & 0xFF;
            case 12: return (pre_x - operand) & 0xFF;
            case 13: return (pre_y - operand) & 0xFF;
            case 14: return operand;
        }
        return 0;
    }"""

IN_RANGES_CPP = """\
    static bool in_ranges(const std::vector<std::pair<int,int>>& v, int addr) {
        for (const auto& r : v) if (addr >= r.first && addr <= r.second) return true;
        return false;
    }"""


def run_state_members_cpp() -> str:
    """Persistent-machine state of the run loop: RAM mirror, architectural
    registers, pause/lease flags, counters, watch ranges, and the
    shadow-decode / trace tables.  Timing is not among them: every
    instruction's cycles are an output of the model that executes it."""
    read_mode = ", ".join(str(v) for v in read_mode_table())
    trace_mode = ", ".join(str(v) for v in trace_mode_table())
    trace_length = ", ".join(str(v) for v in trace_length_table())
    trace_data = ", ".join(str(v) for v in trace_data_table())
    ends_window = ", ".join(str(v) for v in ends_window_table())
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
    static constexpr unsigned char READ_MODE[256] = {{ {read_mode} }};
    static constexpr unsigned char TRACE_MODE[256] = {{ {trace_mode} }};
    static constexpr unsigned char TRACE_LENGTH[256] = {{ {trace_length} }};
    static constexpr unsigned char TRACE_DATA[256] = {{ {trace_data} }};
    // Block entries: single-instruction dispatches not reached by falling
    // through from the previous one (after a control transfer, a window,
    // an interrupt or a pause) -- the heads per-ROM windows grow from.
    static constexpr unsigned char ENDS_WINDOW[256] = {{ {ends_window} }};
    long block_entries[65536] = {{}};
    long branch_runs[65536] = {{}}, branch_taken[65536] = {{}};
    int fallthrough_pc = -1;"""


MACHINE_API_CPP = """\
    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void watch_write(int lo, int hi) {
        wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF);
        if ((lo & 0xFFFF) < watch_write_lo) watch_write_lo = lo & 0xFFFF;
    }
    void clear_watches() {
        rwatch.clear(); wwatch.clear(); watch_write_lo = 0x10000;
    }
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
    int set_read_lease_bytes(int addr, const unsigned char* v, long n) {
        addr &= 0xFFFF;
        const int s = lease_claim(addr);
        const int evicted = lease_addr[s] == addr ? -1 : lease_addr[s];
        lease_addr[s] = addr;
        lease_n[s] = n < 8 ? (int)n : 8;
        lease_i[s] = 0; lease_served[s] = 0;
        for (int i = 0; i < lease_n[s]; ++i) lease_vals[s][i] = v[i];
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

    // The native machine ABI (machine_abi()): the console's bus calls the
    // run loop and these controls directly, with no Python in between.
    // Keep in step with nes/bindings.cpp:cpu_engine (MACHINE_ABI_VERSION).
    struct MachineRunOut {
        const char* reason;  // "budget", "halt", "trap", "watch_read", "watch_write", "cycles"
        long cycles;
        int read_addr;
        long read_cycles;
        int pause_ir;  // the opcode a watch_write pause stopped on
        const int* hit_addr;
        const int* hit_value;
        long n_hits;  // valid until the next run
    };
    struct MachineAbi {
        void (*run)(void*, long max_cycles, MachineRunOut*);
        int (*read_ram)(void*, int);
        void (*inject_write)(void*, int, int);
        void (*map_region)(void*, int, const unsigned char*, long);
        void (*add_window)(void*, int, int);
        int (*set_read_lease)(void*, int, int);
        int (*set_read_lease_seq)(void*, int, const unsigned char*, long);
        long (*take_lease_served)(void*, int);
        void (*clear_read_lease)(void*);
        void (*clear_read_lease_addr)(void*, int);
        void (*set_nmi)(void*);
        void (*set_irq)(void*);
        void (*clear_irq)(void*);
    };
"""

MACHINE_ABI_VERSION = 1

# The pybind export of the machine ABI: a version, the engine, and a
# static table of plain function pointers over it.
PYBIND_MACHINE_ABI = """        .def("machine_abi", [](ENGINE_T& e) {
            using E = ENGINE_T;
            static const E::MachineAbi abi = {
                +[](void* c, long max_cycles, E::MachineRunOut* out) {
                    auto* e = static_cast<E*>(c);
                    const auto r = e->run_core(LONG_MAX, false, max_cycles, false);
                    *out = {r.reason, r.cycles, r.read_addr, r.read_cycles, e->st_ir,
                            e->hit_addr.data(), e->hit_value.data(), (long)e->hit_addr.size()};
                },
                +[](void* c, int a) { return static_cast<E*>(c)->read_ram(a); },
                +[](void* c, int a, int v) { static_cast<E*>(c)->inject_write(a, v); },
                +[](void* c, int base, const unsigned char* d, long n) {
                    static_cast<E*>(c)->map_bytes(base, d, n);
                },
                +[](void* c, int shape, int head) { static_cast<E*>(c)->add_window(shape, head); },
                +[](void* c, int a, int v) { return static_cast<E*>(c)->set_read_lease(a, v); },
                +[](void* c, int a, const unsigned char* v, long n) {
                    return static_cast<E*>(c)->set_read_lease_bytes(a, v, n);
                },
                +[](void* c, int a) { return static_cast<E*>(c)->take_lease_served(a); },
                +[](void* c) { static_cast<E*>(c)->clear_read_lease(); },
                +[](void* c, int a) { static_cast<E*>(c)->clear_read_lease_addr(a); },
                +[](void* c) { static_cast<E*>(c)->set_nmi(); },
                +[](void* c) { static_cast<E*>(c)->set_irq(); },
                +[](void* c) { static_cast<E*>(c)->clear_irq(); },
            };
            return py::make_tuple(MACHINE_ABI_VERSION, reinterpret_cast<std::uintptr_t>(&e),
                                  reinterpret_cast<std::uintptr_t>(&abi));
        })""".replace("MACHINE_ABI_VERSION", str(MACHINE_ABI_VERSION), 1)


RUN_LOOP_TEMPLATE = """\
    // Run up to max_steps instructions, stopping on halt, a jump-to-self
    // trap, a watched device access the leases cannot serve, or the cycle
    // budget.  The console drives run_core through the machine ABI
    // (nes/bindings.cpp:EngineMachine); run() wraps it as a dict for
    // Python, with the per-step trace arrays when `trace` is set.  The
    // per-run outputs are members, reused across runs: the ABI hands out
    // the hit arrays without copying.
    std::vector<int> t_pc, t_a, t_x, t_y, t_s, t_p, t_ir, t_oplo, t_ophi, t_ea, t_data, t_halted;
    std::vector<long> w_row; std::vector<int> w_addr, w_value;
    std::vector<int> hit_addr, hit_value;
    struct RunResult {
        const char* reason;
        int read_addr;
        long read_cycles, cycles, steps, forwards;
    };
    RunResult run_core(long max_steps, bool stop_on_trap, long max_cycles, bool trace) {
        for (auto* v : {&t_pc, &t_a, &t_x, &t_y, &t_s, &t_p, &t_ir, &t_oplo, &t_ophi, &t_ea,
                        &t_data, &t_halted, &w_addr, &w_value, &hit_addr, &hit_value})
            v->clear();
        w_row.clear();
        const char* reason = "budget";
        int read_addr = -1;
        long read_cycles = 0;
        long cycles_this = 0;
        const long steps_entry = step_index;
        long n_forward = 0;
{EXTRA_LOCALS}
        fallthrough_pc = -1;  // after a pause, the next instruction starts a block
        for (long i = 0; i < max_steps; ++i) {
            if (st_halted) { reason = "halt"; break; }
            const bool skip = resume_read_skip;
            resume_read_skip = false;
            if (!nmi_pending && !irq_ready() && !skip && !rwatch.empty()) {
                const int target = shadow_read_target();
                if (target >= 0 && in_ranges(rwatch, target)) {
                    // Leased: serve the device read here, the value the
                    // device would return.  An exhausted queue keeps its
                    // slot (preserving the served count for the bus) but
                    // pauses.
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
                        // The device access lands on the instruction's
                        // last cycle: report its length for the stamp.
                        read_cycles = probe_cycles();
                        resume_read_skip = true; read_addr = target; reason = "watch_read"; break;
                    }
                }
            }
            ++n_forward;
{STEP_BODY}
        }
        return {reason, read_addr, read_cycles, cycles_this, step_index - steps_entry, n_forward};
    }
    py::dict run(long max_steps, bool stop_on_trap, long max_cycles, bool trace) {
        const RunResult r = run_core(max_steps, stop_on_trap, max_cycles, trace);
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
        out["read_addr"] = r.read_addr;
        out["read_cycles"] = r.read_cycles;
        out["reason"] = r.reason;
        out["cycles"] = r.cycles;
        out["steps"] = r.steps;
        out["forwards"] = r.forwards;
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
