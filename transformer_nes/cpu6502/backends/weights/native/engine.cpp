// CPU runtime template consumed by engine.py. Model-derived forwards,
// dispatch and opcode tables fill the marked placeholders.
// CPU scheduling, embedding and readout; neural layers run in Torch.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace py = pybind11;

@MACHINE_ABI@

struct ENGINE_T {
    static constexpr bool MIRROR_CPU_RAM = @MIRROR_CPU_RAM@;
    static constexpr int D = @D_MODEL@;
    long seq = -1;
    static constexpr int F = 1;
    bool xray_on = false;  // count memory reads (xray_reads)
    unsigned int xray_reads[65536] = {};
    static constexpr int SCALE = 1 << F;
    int xbuf[@XBUF_SIZE@];

    static long long shr2(int value) {
        if (value & 1) throw std::runtime_error("fractional readout value");
        return value >> 1;
    }

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
    int lease_addr[NLEASE] = {-1, -1, -1, -1};
    int lease_value[NLEASE] = {0, 0, 0, 0};
    int lease_vals[NLEASE][8];
    int lease_n[NLEASE] = {0, 0, 0, 0};
    int lease_i[NLEASE] = {0, 0, 0, 0};
    long lease_served[NLEASE] = {0, 0, 0, 0};
    long step_index = 0;
    long cycles_total = 0;
    std::vector<std::pair<int,int>> rwatch, wwatch;
    static constexpr unsigned char READ_MODE[256] = { @READ_MODE@ };
    static constexpr unsigned char TRACE_MODE[256] = { @TRACE_MODE@ };
    static constexpr unsigned char TRACE_LENGTH[256] = { @TRACE_LENGTH@ };
    static constexpr unsigned char TRACE_DATA[256] = { @TRACE_DATA@ };


    ENGINE_T() {
        init_shape_runs();
        reset();
    }

    void reset() {
        for (int i = 0; i < 65536; ++i) ram[i] = 0;
        std::memset(shape_of_pc, 0, sizeof(shape_of_pc));
        st_pc = 0; st_a = st_x = st_y = 0; st_s = 0xFD; st_p = 0x24;
        st_ir = st_oplo = st_ophi = st_ea = st_data = 0;
        st_halted = false; nmi_pending = false; irq_pending = false;
        resume_read_skip = false;
        clear_read_lease();
        step_index = 0; cycles_total = 0;
        clear_watches();
        seq = -1;
    }


    void insert_write(int addr, int value, long pos) {
        ++seq;
        if (seq != pos)
            throw std::runtime_error("direct WRITE position does not match engine sequence");
        addr &= 0xFFFF; value &= 0xFF;
        const int first = (MIRROR_CPU_RAM && addr < 0x2000) ? (addr & 0x07FF) : addr;
        const int last = (MIRROR_CPU_RAM && addr < 0x2000) ? first + 0x1800 : first;
        for (int mapped = first; mapped <= last; mapped += 0x0800) {
            ram[mapped] = (unsigned char)value;  // shadow-decode mirror
        }
    }

    // One whole 6502 instruction from the persistent st_* state:
    // STATE embed -> layers -> readout -> direct WRITE inserts ->
    // halt state.  Fills the 22-slot out buffer (trace row, next
    // state, writes, cycles) and advances st_* to the next instruction.

@FORWARDS@

@DISPATCH@

    // The cycles the last expert forward retired: its model's output.
    long seq_cycles = 0;
    // Required Torch bank entry over the embedded residual stream.
    // IDs follow the core expert order; a missing entry stops execution.
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

    // Instructions retired per execution path since the last xray_take():
    // shape experts, interrupt experts, opcode experts,
    // the union model; the dispatches (forwards, or a whole spin or loop)
    // that retired them; and per opcode, those that ran singly.
    long xray_paths[4] = {};
    long xray_dispatches[4] = {};
    long xray_opcodes[256] = {};

    enum { LOOP_NONE, LOOP_POLL, LOOP_COUNTER };
    struct Loop { int kind, closure_offset, closure_value, reg, delta; };
    static Loop read_loop(const int* p) { return {p[0], p[1], p[2], p[3], p[4]}; }
    struct ShapeRun {
        int steps, prefix, spin;
        Loop loop;
    };
    bool loop_closes(const Loop& l, int head) const {
        return st_pc == head && ram[(head + l.closure_offset) & 0xFFFF] == l.closure_value;
    }
    // Retire only complete iterations that individual instructions would run
    // within these budgets. A final partial iteration stays on the step path.
    template <class Forward>
    int run_shape(const ShapeRun& w, long i, long max_steps, long cycles_this, long max_cycles,
                  Forward&& forward) {
        const long steps = w.spin ? 1 : w.steps;
        if (i + steps > max_steps || (max_cycles > 0 && cycles_this + (w.spin ? 0 : w.prefix) >= max_cycles))
            return 0;
        if (w.spin) {
            long n = max_steps - i;
            if (max_cycles > 0) n = std::min(n, (max_cycles - cycles_this - 1) / w.spin + 1);
            seq_cycles = n * w.spin;
            return (int)n;
        }
        const int head = st_pc;
        const int before = w.loop.reg == 1 ? st_x : st_y;
        forward();
        if (w.loop.kind == LOOP_NONE || !loop_closes(w.loop, head)) return (int)steps;
        long extra = (max_steps - i) / steps - 1;
        if (max_cycles > 0)
            extra = std::min(extra, (max_cycles - cycles_this - w.prefix - 1) / seq_cycles);
        if (w.loop.kind == LOOP_POLL) {
            const int polled = shadow_read_target();
            const int lease = lease_slot(polled);
            const bool holds = !in_ranges(rwatch, polled)
                || (lease >= 0 && lease_n[lease] == 0 && lease_value[lease] == ram[polled]);
            if (!holds) extra = 0;
        } else {
            int& counter = w.loop.reg == 1 ? st_x : st_y;
            if (((counter - before) & 0xFF) != (w.loop.delta & 0xFF) || !counter) return (int)steps;
            // Stop at the last nonzero result. The zero/fallthrough iteration
            // runs through the transformer, including its different cycles.
            extra = std::min(extra, (long)(w.loop.delta < 0 ? counter - 1 : 255 - counter));
            counter = (counter + extra * w.loop.delta) & 0xFF;
            st_p = (st_p & ~0x82) | (counter & 0x80);
            seq += extra;
        }
        seq_cycles *= 1 + extra;
        return (int)((1 + extra) * steps);
    }

    std::vector<ShapeRun> shape_runs;
    void init_shape_runs() {
@SHAPE_RUNS@
    }


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
    }

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
    }

    static bool in_ranges(const std::vector<std::pair<int,int>>& v, int addr) {
        for (const auto& r : v) if (addr >= r.first && addr <= r.second) return true;
        return false;
    }

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
    }

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
        long long o[22];
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
            // A fixed shape consumes several instructions in one forward;
            // its byte and budget guards keep this step-for-step
            // equivalent to the per-instruction path below.
            if (!trace) {
                const int k_seq = shape_dispatch(i, max_steps, cycles_this,
                                               max_cycles, stop_on_trap);
                if (k_seq > 0) {
                    xray_paths[0] += k_seq;
                    ++xray_dispatches[0];
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
                ++xray_paths[was_int ? 1 : 2];
                ++xray_dispatches[was_int ? 1 : 2];
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
            ++xray_paths[3];
            ++xray_dispatches[3];
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
            if (max_cycles > 0 && cycles_this >= max_cycles) { reason = "cycles"; break; }
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
    }

    // ── Machine API (machine_abi.h) ──────────────────────────

    void watch_read(int lo, int hi) { rwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF); }
    void watch_write(int lo, int hi) {
        wwatch.emplace_back(lo & 0xFFFF, hi & 0xFFFF);
    }
    void clear_watches() {
        rwatch.clear(); wwatch.clear();
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

    using MachineRunOut = cpu_engine::RunOut;
    using MachineAbi = cpu_engine::Abi;


    void inject_write(int addr, int value) { insert_write(addr & 0xFFFF, value & 0xFF, seq + 1); }

    // Cartridge mapping changes address translation; it does not append
    // thousands of fictional CPU WRITE tokens.  All 6502 memory heads use
    // this exact-address table, so replacing a mapped byte in place preserves
    // reads while leaving transformer sequence position unchanged.
    void map_write(int addr, int value) {
        addr &= 0xFFFF; value &= 0xFF;
        ram[addr] = (unsigned char)value;
    }

    void map_bytes(int base, const unsigned char* d, long n) {
        for (long i = 0; i < n; ++i)
            map_write(base + (int)i, d[i]);
    }
    void map_region(int base, py::bytes data) {
        const std::string d = data;
        map_bytes(base, (const unsigned char*)d.data(), (long)d.size());
    }

    // Seed the full image, recording nonzero physical bytes as WRITE tokens.
    // Mirrored RAM is delivered by insert_write; return the tokens added.
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
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ENGINE_T>(m, "Engine", py::module_local())
        .def(py::init<>())
        .def("reset", &ENGINE_T::reset)
        .def("run", &ENGINE_T::run, py::arg("max_steps"),
             py::arg("stop_on_trap") = false, py::arg("max_cycles") = 0L,
             py::arg("trace") = true)
        .def("watch_read", &ENGINE_T::watch_read)
        .def("watch_write", &ENGINE_T::watch_write)
        .def("set_nmi", &ENGINE_T::set_nmi)
        .def("set_irq", &ENGINE_T::set_irq)
        .def("clear_irq", &ENGINE_T::clear_irq)
        .def("map_region", &ENGINE_T::map_region)
        .def("seed_image", &ENGINE_T::seed_image)
        .def("read_ram", &ENGINE_T::read_ram)
        .def("dump_ram", &ENGINE_T::dump_ram)
        .def("set_run_state", &ENGINE_T::set_run_state)
        .def("get_run_state", &ENGINE_T::get_run_state)
        .def("set_forward_hook", &ENGINE_T::set_forward_hook)
        .def("ram_address", &ENGINE_T::ram_address)
        .def("xray_address", &ENGINE_T::xray_address)
        .def("add_window", &ENGINE_T::add_window)
        .def("clear_windows", &ENGINE_T::clear_windows)
        .def("machine_abi", [](ENGINE_T& e) {
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
            return py::make_tuple(cpu_engine::abi_version, reinterpret_cast<std::uintptr_t>(&e),
                                  reinterpret_cast<std::uintptr_t>(&abi));
        })
        .def("xray_take", [](ENGINE_T& e) {
            py::list paths;
            for (long& n : e.xray_paths) paths.append(std::exchange(n, 0));
            py::list dispatches;
            for (long& n : e.xray_dispatches) dispatches.append(std::exchange(n, 0));
            py::array_t<long> ops(256);
            for (int i = 0; i < 256; ++i) ops.mutable_data()[i] = std::exchange(e.xray_opcodes[i], 0);
            return py::make_tuple(paths, ops, dispatches);
        })
        .def("xray_reads", [](ENGINE_T& e, bool on) {
            py::array_t<unsigned int> out(65536);
            std::memcpy(out.mutable_data(), e.xray_reads, sizeof e.xray_reads);
            std::memset(e.xray_reads, 0, sizeof e.xray_reads);
            e.xray_on = on;
            return out;
        })
        .def("union_inputs", [](ENGINE_T& e) {
            e.embed_union();
            py::array_t<int> out(@D_MODEL@);
            std::memcpy(out.mutable_data(), e.xbuf, sizeof(int) * out.size());
            return out;
        })
        .def("inject_write", &ENGINE_T::inject_write);
}
