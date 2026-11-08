#include "nes/cpu.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace transformer_nes::nes {

namespace {

// cpu6502/chip.py INPUTS: the state, instruction context and read, the
// interrupt lines, loop bounds, P's flags and A's bits (MSB first).
enum Input {
  in_state, in_pc_lo, in_pc_hi, in_a, in_x, in_y, in_s, in_ir, in_oplo, in_ophi, in_ea_lo, in_ea_hi,
  in_data, in_halted, in_nmi, in_irq, in_read_lo, in_read_hi, in_read_cycles, in_loop_limit,
  in_loop_cycles, in_flags, in_a_bits = in_flags + 8,
};
constexpr std::array<int, 8> flag_masks{0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
// STEP_OUTPUTS: registers, flags (C Z I D B V N), then the step's effects.
enum StepOutput {
  out_pc, out_a, out_x, out_y, out_s, out_flags, out_halted = out_flags + 7, out_cycles,
  out_iterations, out_writes, out_write,
};
// Each step flag's place among P's input fields (C Z I D B U V N).
constexpr std::array<size_t, 7> step_fields{0, 1, 2, 3, 4, 6, 7};
// UNION_OUTPUTS (graph.py RUNTIME_READ_DIMS).
enum UnionOutput {
  u_current_pc, u_a, u_x, u_y, u_s, u_p, u_halted, u_opcode, u_oplo, u_ophi, u_ea, u_data, u_next_pc,
  u_store, u_cycles = u_store + 9, u_original,
};

} // namespace

void NeuralCpu::Watches::add(int lo, int hi) {
  lo &= 65535;
  hi &= 65535;
  if (lo > hi)
    return;
  const int first = lo >> 6, last = hi >> 6;
  const auto head = ~std::uint64_t{0} << (lo & 63);
  const auto tail = ~std::uint64_t{0} >> (63 - (hi & 63));
  if (first == last) {
    words[first] |= head & tail;
  } else {
    words[first] |= head;
    std::fill(words.begin() + first + 1, words.begin() + last, ~std::uint64_t{0});
    words[last] |= tail;
  }
  any = true;
}

void NeuralCpu::Trace::clear() {
  for (auto* v : {&pc, &a, &x, &y, &s, &p, &ir, &oplo, &ophi, &ea, &data, &halted, &w_addr,
                  &w_value, &w_original})
    v->clear();
  w_row.clear();
}

NeuralCpu::NeuralCpu(void* engine, Step step, std::uint8_t* memory, CpuTables tables,
                     bool mirror_cpu_ram)
    : engine_(engine), step_(step), ram_(memory), tables_(std::move(tables)),
      mirror_cpu_ram_(mirror_cpu_ram), shape_of_pc_(65536) {
  reset();
}

void NeuralCpu::rebind(void* engine, Step step, std::uint8_t* memory) {
  if (memory != ram_)
    std::memcpy(memory, ram_, 65536);
  engine_ = engine;
  step_ = step;
  ram_ = memory;
}

void NeuralCpu::reset() {
  std::memset(ram_, 0, 65536);
  clear_windows();
  in_.fill(0);
  pc_ = ea_ = 0;
  set_a(0);
  s_ = 0xFD;
  set_p(0x24);
  nmi_pending_ = irq_pending_ = resume_read_skip_ = false;
  pending_read_ = {};
  clear_read_lease();
  step_index_ = cycles_total_ = 0;
  clear_watches();
}

// ── One circuit's forward over the state ──────────────────────────

int* NeuralCpu::embed(int model, bool consume) {
  int* in = in_.data();
  in[in_state] = 1;
  in[in_pc_lo] = pc_ & 0xFF;
  in[in_pc_hi] = (pc_ >> 8) & 0xFF;
  in[in_ea_lo] = ea_ & 0xFF;
  in[in_ea_hi] = (ea_ >> 8) & 0xFF;
  in[in_nmi] = in[in_irq] = 0;
  if (tables_.reads_interrupts[static_cast<size_t>(model)]) {
    // A circuit that reads the lines takes what it is shown.
    const bool take_nmi = nmi_pending_ && pending_read_.address < 0;
    const bool take_irq = irq_ready() && pending_read_.address < 0;
    in[in_nmi] = take_nmi;
    in[in_irq] = take_irq;
    if (consume) {
      if (take_nmi)
        nmi_pending_ = false;
      if (take_irq)
        irq_pending_ = false;
    }
  }
  in[in_read_lo] = pending_read_.address & 0xFF;
  in[in_read_hi] = (pending_read_.address >> 8) & 0xFF;
  in[in_read_cycles] = pending_read_.cycles;
  in[in_loop_limit] = loop_limit_;
  in[in_loop_cycles] = loop_cycles_;
  return in;
}

int NeuralCpu::p() const {
  int p = 0;
  for (int f = 0; f < 8; ++f)
    p |= in_[static_cast<size_t>(in_p + f)] ? flag_masks[static_cast<size_t>(f)] : 0;
  return p;
}

void NeuralCpu::set_p(int p) {
  for (int f = 0; f < 8; ++f)
    in_[static_cast<size_t>(in_p + f)] = (p & flag_masks[static_cast<size_t>(f)]) ? 1 : 0;
}

void NeuralCpu::set_a(int a) {
  a_ = a;
  for (int b = 0; b < 8; ++b)
    in_[static_cast<size_t>(in_a_bits + b)] = (a >> (7 - b)) & 1;
}

// A step circuit: registers, flags and halting from its outputs; its
// writes staged for the run loop (an instruction) or made (a shape).
void NeuralCpu::step(int model, bool stage_writes) {
  int out[step_outputs];
  forward(model, embed(model), out);
  pc_ = out[out_pc] & 0xFFFF;
  set_a(out[out_a] & 0xFF);
  x_ = out[out_x] & 0xFF;
  y_ = out[out_y] & 0xFF;
  s_ = out[out_s] & 0xFF;
  for (size_t f = 0; f < step_fields.size(); ++f)
    in_[static_cast<size_t>(in_p) + step_fields[f]] = out[out_flags + static_cast<int>(f)] & 1;
  halted_ = out[out_halted] & 1;
  seq_cycles_ = out[out_cycles];
  loop_iterations_ = out[out_iterations];
  n_writes_ = 0;
  for (int w = 0; w < out[out_writes]; ++w) {
    const int* write = out + out_write + 3 * w;
    if (stage_writes) {
      write_addr_[static_cast<size_t>(w)] = write[0] & 0xFFFF;
      write_value_[static_cast<size_t>(w)] = write[1] & 0xFF;
      write_original_[static_cast<size_t>(w)] = write[2];
      ++n_writes_;
    } else {
      this->write(write[0], write[1]);
    }
  }
}

bool NeuralCpu::execution_step(int opcode, bool read_serviced) {
  n_writes_ = 0;
  const int model = tables_.exec[static_cast<size_t>(opcode)];
  if (model == CpuTables::none || (tables_.exec_after_read[static_cast<size_t>(opcode)] && !read_serviced))
    return false;
  step(model, true);
  return true;
}

bool NeuralCpu::interrupt_step(bool nmi) {
  n_writes_ = 0;
  const int model = nmi ? tables_.nmi : tables_.irq;
  if (model == CpuTables::none)
    return false;
  step(model, true);
  return true;
}

NeuralCpu::ReadRequest NeuralCpu::request_read() {
  read_ir_ = ram_[pc_];
  const int model = tables_.request[static_cast<size_t>(read_ir_)];
  if (model == CpuTables::none)
    return {};
  int out[request_outputs];
  forward(model, embed(model), out);
  return {out[0], out[1], out[2], out[3]};
}

void NeuralCpu::counter_recurrence(int reg, int delta, int limit, int cycles) {
  loop_limit_ = limit;
  loop_cycles_ = cycles;
  for (const auto& loop : tables_.loops)
    if (loop.reg == reg && loop.delta == delta) {
      step(loop.model, false);
      return;
    }
  throw std::runtime_error("counter recurrence transformer is missing");
}

// The union circuit: any byte or interrupt. Fills the receipt o[] (trace
// row, next state, writes, cycles) and advances the state.
void NeuralCpu::union_step(long long* o) {
  int u[union_outputs];
  forward(tables_.union_model, embed(tables_.union_model), u);
  o[0] = u[u_current_pc] & 0xFFFF;
  o[1] = u[u_a] & 0xFF;
  o[2] = u[u_x] & 0xFF;
  o[3] = u[u_y] & 0xFF;
  o[4] = u[u_s] & 0xFF;
  o[5] = u[u_p] & 0xFF;
  o[6] = u[u_opcode] & 0xFF;
  o[7] = u[u_oplo] & 0xFF;
  o[8] = u[u_ophi] & 0xFF;
  o[9] = u[u_ea] & 0xFFFF;
  o[10] = u[u_data] & 0xFF;
  o[11] = u[u_halted] & 0xFF;
  o[12] = u[u_next_pc] & 0xFFFF;
  long long n = 0;
  for (int w = 0; w < 3; ++w) {
    const int* store = u + u_store + 3 * w;
    if ((store[0] & 0xFF) != 0) {
      o[15 + 2 * n] = store[1] & 0xFFFF;
      o[16 + 2 * n] = store[2] & 0xFF;
      ++n;
    }
  }
  o[14] = n;
  o[21] = u[u_cycles] & 0xFF;
  o[22] = u[u_original];
  for (long long i = 0; i < n; ++i)
    write(static_cast<int>(o[15 + 2 * i]), static_cast<int>(o[16 + 2 * i]));
  o[13] = o[11] != 0;
  pc_ = static_cast<int>(o[12]);
  set_a(static_cast<int>(o[1]));
  x_ = static_cast<int>(o[2]);
  y_ = static_cast<int>(o[3]);
  s_ = static_cast<int>(o[4]);
  set_p(static_cast<int>(o[5]));
  ir_ = static_cast<int>(o[6]);
  oplo_ = static_cast<int>(o[7]);
  ophi_ = static_cast<int>(o[8]);
  ea_ = static_cast<int>(o[9]);
  data_ = static_cast<int>(o[10]);
  halted_ = o[11] != 0;
}

std::array<int, NeuralCpu::inputs> NeuralCpu::union_inputs() {
  const int* in = embed(tables_.union_model, false);
  std::array<int, inputs> out{};
  std::copy_n(in, inputs, out.begin());
  return out;
}

// ── Memory ────────────────────────────────────────────────────────

void NeuralCpu::write(int addr, int value) {
  addr &= 0xFFFF;
  value &= 0xFF;
  const int first = (mirror_cpu_ram_ && addr < 0x2000) ? (addr & 0x07FF) : addr;
  const int last = (mirror_cpu_ram_ && addr < 0x2000) ? first + 0x1800 : first;
  for (int mapped = first; mapped <= last; mapped += 0x0800)
    ram_[mapped] = static_cast<std::uint8_t>(value);
}

void NeuralCpu::inject_write(int addr, int value) { write(addr, value); }

// Cartridge mapping replaces bytes in place: no CPU write happens.
void NeuralCpu::map_bytes(int base, const std::uint8_t* data, long n) {
  base &= 0xFFFF;
  while (n > 0) {
    const long run = std::min(n, 0x10000L - base);
    std::memcpy(ram_ + base, data, static_cast<size_t>(run));
    data += run;
    n -= run;
    base = 0;
  }
}

void NeuralCpu::seed_image(const std::uint8_t* image) {
  for (int a = 0; a < 65536; ++a)
    if (!(mirror_cpu_ram_ && a >= 0x0800 && a < 0x2000))
      write(a, image[a]);
}

void NeuralCpu::set_state(int pc, int a, int x, int y, int s, int p, int ir, int oplo, int ophi,
                          int ea, int data, bool halted) {
  pc_ = pc & 0xFFFF;
  set_a(a & 0xFF);
  x_ = x & 0xFF;
  y_ = y & 0xFF;
  s_ = s & 0xFF;
  set_p((p | 0x20) & 0xFF);
  ir_ = ir & 0xFF;
  oplo_ = oplo & 0xFF;
  ophi_ = ophi & 0xFF;
  ea_ = ea & 0xFFFF;
  data_ = data & 0xFF;
  halted_ = halted;
}

NeuralCpu::State NeuralCpu::state() const {
  return {pc_, a_, x_, y_, s_, p(), ir_, oplo_, ophi_, ea_, data_, halted_ != 0, step_index_, cycles_total_};
}

// ── Shapes: fixed windows of instructions run as one forward ─────

void NeuralCpu::add_window(int shape, int head) {
  if (shape < 0 || shape >= static_cast<int>(tables_.shapes.size()))
    throw std::runtime_error("unknown shape id");
  if (head < 0 || head + tables_.shapes[static_cast<size_t>(shape)].length > 0x10000)
    throw std::runtime_error("shape window would wrap $FFFF");
  shape_of_pc_[static_cast<size_t>(head)] = static_cast<std::uint8_t>(shape + 1);
}

void NeuralCpu::clear_windows() { std::fill(shape_of_pc_.begin(), shape_of_pc_.end(), 0); }

int NeuralCpu::shape_dispatch(long i, long max_steps, long cycles_this, long max_cycles,
                              bool stop_on_trap) {
  const int sid = shape_of_pc_[static_cast<size_t>(pc_)];
  if (!sid || nmi_pending_ || irq_ready() || stop_on_trap)
    return 0;
  // A device reply may have changed instruction bytes. The first
  // instruction must still match its retained prefix.
  if (pending_read_.address >= 0 &&
      (ram_[pc_] != read_ir_ || ram_[pc_ + 1] != pending_read_.oplo ||
       (tables_.trace_length[static_cast<size_t>(read_ir_)] > 2 && ram_[pc_ + 2] != pending_read_.ophi)))
    return 0;
  const auto& shape = tables_.shapes[static_cast<size_t>(sid - 1)];
  // add_window guarantees head + length <= 0x10000: no wrap.
  for (const auto& [offset, opcode] : shape.checks)
    if (ram_[pc_ + offset] != opcode)
      return 0;
  if (shape.self_target && (ram_[pc_ + 1] != (pc_ & 0xFF) || ram_[pc_ + 2] != ((pc_ >> 8) & 0xFF)))
    return 0;
  return run_shape(shape, i, max_steps, cycles_this, max_cycles);
}

// Retire only complete iterations that individual instructions would run
// within these budgets. A final partial iteration stays on the step path.
int NeuralCpu::run_shape(const CpuTables::Shape& w, long i, long max_steps, long cycles_this,
                         long max_cycles) {
  const long steps = w.spin ? 1 : w.steps;
  if (i + steps > max_steps || (max_cycles > 0 && cycles_this + (w.spin ? 0 : w.prefix) >= max_cycles))
    return 0;
  if (w.spin) {
    long n = max_steps - i;
    if (max_cycles > 0)
      n = std::min(n, (max_cycles - cycles_this - 1) / w.spin + 1);
    seq_cycles_ = n * w.spin;
    return static_cast<int>(n);
  }
  const int head = pc_;
  const int kind = w.loop[0], closure_offset = w.loop[1], closure_value = w.loop[2],
            reg = w.loop[3], delta = w.loop[4];
  const int before = reg == 1 ? x_ : y_;
  step(w.model, false);
  if (kind == LOOP_NONE || !(pc_ == head && ram_[(head + closure_offset) & 0xFFFF] == closure_value))
    return static_cast<int>(steps);
  long extra = (max_steps - i) / steps - 1;
  if (max_cycles > 0)
    extra = std::min(extra, (max_cycles - cycles_this - w.prefix - 1) / seq_cycles_);
  if (kind == LOOP_POLL && !rwatch_.empty()) {
    const int polled = pending_read_.address;
    const int lease = lease_slot(polled);
    const bool holds = !rwatch_.contains(polled) ||
                       (lease >= 0 && lease_n_[static_cast<size_t>(lease)] == 0 &&
                        lease_value_[static_cast<size_t>(lease)] == ram_[polled]);
    if (!holds)
      extra = 0;
  } else if (kind == LOOP_COUNTER) {
    int& counter = reg == 1 ? x_ : y_;
    if (((counter - before) & 0xFF) != (delta & 0xFF) || !counter)
      return static_cast<int>(steps);
    // Stop at the last nonzero result. The zero/fallthrough iteration
    // runs through the transformer, including its different cycles.
    if (extra <= 0)
      return static_cast<int>(steps);
    const long initial_cycles = seq_cycles_;
    counter_recurrence(reg, delta, static_cast<int>(std::min(extra, 255L)), static_cast<int>(initial_cycles));
    seq_cycles_ += initial_cycles;
    return static_cast<int>((1 + loop_iterations_) * steps);
  }
  seq_cycles_ *= 1 + extra;
  return static_cast<int>((1 + extra) * steps);
}

// ── Traces ────────────────────────────────────────────────────────

int NeuralCpu::trace_ea(int opcode, int pc, int xr, int yr) const {
  const int lo = ram_[(pc + 1) & 0xFFFF];
  const int hi = ram_[(pc + 2) & 0xFFFF];
  switch (tables_.trace_mode[static_cast<size_t>(opcode)]) {
  case 1:
  case 11:
    return lo;
  case 2:
    return (lo + xr) & 0xFF;
  case 3:
    return (lo + yr) & 0xFF;
  case 4:
    return lo | (hi << 8);
  case 5:
    return ((lo | (hi << 8)) + xr) & 0xFFFF;
  case 6:
    return ((lo | (hi << 8)) + yr) & 0xFFFF;
  case 7: {
    const int zp = (lo + xr) & 0xFF;
    return ram_[zp] | (ram_[(zp + 1) & 0xFF] << 8);
  }
  case 8: {
    const int base = ram_[lo] | (ram_[(lo + 1) & 0xFF] << 8);
    return (base + yr) & 0xFFFF;
  }
  case 9: {
    const int ptr = lo | (hi << 8);
    const int next = (ptr & 0xFF00) | ((ptr + 1) & 0xFF);
    return ram_[ptr] | (ram_[next] << 8);
  }
  case 10:
    return (pc + 2 + static_cast<int>(static_cast<std::int8_t>(lo))) & 0xFFFF;
  }
  return 0;
}

int NeuralCpu::trace_data(int opcode, int pre_a, int pre_x, int pre_y, int pre_p,
                          int operand) const {
  switch (tables_.trace_data[static_cast<size_t>(opcode)]) {
  case 1: return pre_a;
  case 2: return pre_x;
  case 3: return pre_y;
  case 4: return pre_p;
  case 5: return write_value_[0];
  case 6: return a_;
  case 7: return x_;
  case 8: return y_;
  case 9: return s_;
  case 10: return p();
  case 11: return (pre_a - operand) & 0xFF;
  case 12: return (pre_x - operand) & 0xFF;
  case 13: return (pre_y - operand) & 0xFF;
  case 14: return operand;
  }
  return 0;
}

void NeuralCpu::trace_row(int pc, int a, int x, int y, int s, int p, int ir, int oplo, int ophi,
                          int ea, int data, bool halted) {
  auto& t = trace_;
  t.pc.push_back(pc);
  t.a.push_back(a);
  t.x.push_back(x);
  t.y.push_back(y);
  t.s.push_back(s);
  t.p.push_back(p);
  t.ir.push_back(ir);
  t.oplo.push_back(oplo);
  t.ophi.push_back(ophi);
  t.ea.push_back(ea);
  t.data.push_back(data);
  t.halted.push_back(halted ? 1 : 0);
}

// An instruction or interrupt's staged writes, in order: into memory, the
// trace and the watched-write hits. Returns the hits.
int NeuralCpu::commit_writes(bool trace) {
  int hits = 0;
  for (int w = 0; w < n_writes_; ++w) {
    const auto k = static_cast<size_t>(w);
    const int wa = write_addr_[k], wv = write_value_[k];
    if (trace) {
      trace_.w_row.push_back(step_index_);
      trace_.w_addr.push_back(wa);
      trace_.w_value.push_back(wv);
      trace_.w_original.push_back(write_original_[k]);
    }
    write(wa, wv);
    if (!wwatch_.empty() && wwatch_.contains(wa)) {
      hits_.addr.push_back(wa);
      hits_.value.push_back(wv);
      hits_.original.push_back(write_original_[k]);
      ++hits;
    }
  }
  return hits;
}

// ── The run loop ──────────────────────────────────────────────────

NeuralCpu::Run NeuralCpu::run(long max_steps, bool stop_on_trap, long max_cycles, bool trace) {
  trace_.clear();
  hits_ = {};
  const char* reason = "budget";
  int read_addr = -1;
  long read_cycles = 0, cycles_this = 0, n_forward = 0;
  const long steps_entry = step_index_;
  long long o[23];
  for (long i = 0; i < max_steps; ++i) {
    if (halted_) {
      reason = "halt";
      break;
    }
    const bool skip = resume_read_skip_;
    resume_read_skip_ = false;
    bool read_serviced = skip;
    if (!nmi_pending_ && !irq_ready() && !skip) {
      const auto request = pending_read_ = request_read();
      const int target = request.address;
      if (target >= 0 && !rwatch_.empty() && rwatch_.contains(target)) {
        // Leased: serve the device read here, the value the device would
        // return. An exhausted queue keeps its slot (preserving the served
        // count for the bus) but pauses.
        bool served = false;
        const int ls = lease_slot(target);
        if (ls >= 0) {
          const auto s = static_cast<size_t>(ls);
          if (lease_n_[s] == 0) {
            write(target, lease_value_[s]);
            served = true;
          } else if (lease_i_[s] < lease_n_[s]) {
            write(target, lease_values_[s][static_cast<size_t>(lease_i_[s]++)]);
            ++lease_served_[s];
            served = true;
          }
        }
        read_serviced = served;
        if (!served) {
          // The device access lands on the instruction's last cycle:
          // report its length for the stamp.
          read_cycles = request.cycles;
          resume_read_skip_ = true;
          read_addr = target;
          reason = "watch_read";
          break;
        }
      }
    }
    ++n_forward;
    // A fixed shape consumes several instructions in one forward; its byte
    // and budget guards keep this step-for-step equivalent to single steps.
    if (!trace) {
      const int k_seq = shape_dispatch(i, max_steps, cycles_this, max_cycles, stop_on_trap);
      if (k_seq > 0) {
        pending_read_.address = -1;
        xray_paths[0] += k_seq;
        ++xray_dispatches[0];
        step_index_ += k_seq;
        cycles_this += seq_cycles_;
        cycles_total_ += seq_cycles_;
        if (halted_) {
          reason = "halt";
          break;
        }
        i += k_seq - 1;
        if (max_cycles > 0 && cycles_this >= max_cycles) {
          reason = "cycles";
          break;
        }
        continue;
      }
    }
    const bool was_int = !skip && (nmi_pending_ || irq_ready());
    const int prev_pc = pc_;
    const bool read = !was_int && pending_read_.address >= 0;
    const int exec_opcode = read ? read_ir_ : ram_[prev_pc];
    const int pre_a = a_, pre_x = x_, pre_y = y_, pre_p = trace ? p() : 0;
    int tr_lo = 0, tr_hi = 0, tr_ea = 0, tr_operand = 0;
    if (trace) {
      const int tr_len = tables_.trace_length[static_cast<size_t>(exec_opcode)];
      if (tr_len > 1 || was_int)
        tr_lo = read ? pending_read_.oplo : ram_[(prev_pc + 1) & 0xFFFF];
      if (tr_len > 2 || was_int)
        tr_hi = read ? pending_read_.ophi : ram_[(prev_pc + 2) & 0xFFFF];
      // Interrupts report the interrupted PC with no effective address.
      if (!was_int) {
        tr_ea = read ? pending_read_.address : trace_ea(exec_opcode, prev_pc, pre_x, pre_y);
        if (tables_.trace_data[static_cast<size_t>(exec_opcode)] >= 11)
          tr_operand = tables_.trace_mode[static_cast<size_t>(exec_opcode)] == 11 ? tr_lo : ram_[tr_ea];
      }
    }
    if (was_int ? interrupt_step(nmi_pending_) : execution_step(exec_opcode, read_serviced)) {
      pending_read_.address = -1;
      ++xray_paths[was_int ? 1 : 2];
      ++xray_dispatches[was_int ? 1 : 2];
      if (!was_int)
        ++xray_opcodes[static_cast<size_t>(exec_opcode)];
      const int tr_data =
          trace && !was_int ? trace_data(exec_opcode, pre_a, pre_x, pre_y, pre_p, tr_operand) : 0;
      const int hits = commit_writes(trace);
      if (trace) {
        trace_row(prev_pc, a_, x_, y_, s_, p(), exec_opcode, tr_lo, tr_hi, tr_ea, tr_data, halted_ != 0);
        ir_ = exec_opcode;
        oplo_ = tr_lo;
        ophi_ = tr_hi;
        ea_ = tr_ea;
        data_ = tr_data;
      }
      if (was_int)
        nmi_pending_ = false;
      ++step_index_;
      cycles_this += seq_cycles_;
      cycles_total_ += seq_cycles_;
      if (!was_int && stop_on_trap && pc_ == prev_pc) {
        reason = "trap";
        break;
      }
      if (hits > 0) {
        reason = "watch_write";
        break;
      }
      if (halted_) {
        reason = "halt";
        break;
      }
      if (max_cycles > 0 && cycles_this >= max_cycles) {
        reason = "cycles";
        break;
      }
      continue;
    }
    // The union circuit (undefined bytes, unfolded interrupts): its
    // receipt o[] carries the state, writes and stop flags.
    union_step(o);
    pending_read_.address = -1;
    ++xray_paths[3];
    ++xray_dispatches[3];
    ++xray_opcodes[static_cast<size_t>(exec_opcode)];
    const bool halted = o[11] != 0;
    if (trace)
      trace_row(static_cast<int>(o[0]), static_cast<int>(o[1]), static_cast<int>(o[2]),
                static_cast<int>(o[3]), static_cast<int>(o[4]), static_cast<int>(o[5]),
                static_cast<int>(o[6]), static_cast<int>(o[7]), static_cast<int>(o[8]),
                static_cast<int>(o[9]), static_cast<int>(o[10]), halted);
    int hits = 0;
    for (int w = 0; w < static_cast<int>(o[14]); ++w) {
      const int wa = static_cast<int>(o[15 + 2 * w]), wv = static_cast<int>(o[16 + 2 * w]);
      const int original = w == 0 ? static_cast<int>(o[22]) : -1;
      if (trace) {
        trace_.w_row.push_back(step_index_);
        trace_.w_addr.push_back(wa);
        trace_.w_value.push_back(wv);
        trace_.w_original.push_back(original);
      }
      if (!wwatch_.empty() && wwatch_.contains(wa)) {
        hits_.addr.push_back(wa);
        hits_.value.push_back(wv);
        hits_.original.push_back(original);
        ++hits;
      }
    }
    ++step_index_;
    cycles_this += o[21];
    cycles_total_ += o[21];
    if (halted || o[13]) {
      halted_ = true;
      reason = "halt";
      break;
    }
    if (stop_on_trap && static_cast<int>(o[12]) == prev_pc) {
      reason = "trap";
      break;
    }
    if (hits > 0) {
      reason = "watch_write";
      break;
    }
    if (max_cycles > 0 && cycles_this >= max_cycles) {
      reason = "cycles";
      break;
    }
  }
  return {reason, read_addr, read_cycles, cycles_this, step_index_ - steps_entry, n_forward};
}

// ── Leases ────────────────────────────────────────────────────────

int NeuralCpu::lease_slot(int addr) const {
  for (int s = 0; s < lease_capacity; ++s)
    if (lease_addr_[static_cast<size_t>(s)] == addr)
      return s;
  return -1;
}

// Reuse the address's slot, else a free one, else an exhausted queue, else
// slot 0. The bus flushes served counts before every set, so an eviction
// never loses a device advance.
int NeuralCpu::lease_claim(int addr) {
  int s = lease_slot(addr);
  if (s < 0)
    s = lease_slot(-1);
  if (s < 0)
    for (int t = 0; t < lease_capacity; ++t) {
      const auto k = static_cast<size_t>(t);
      if (lease_n_[k] > 0 && lease_i_[k] == lease_n_[k]) {
        s = t;
        break;
      }
    }
  return s < 0 ? 0 : s;
}

int NeuralCpu::set_read_lease(int addr, int value) {
  addr &= 0xFFFF;
  const auto s = static_cast<size_t>(lease_claim(addr));
  const int evicted = lease_addr_[s] == addr ? -1 : lease_addr_[s];
  lease_addr_[s] = addr;
  lease_value_[s] = value & 0xFF;
  lease_n_[s] = lease_i_[s] = 0;
  lease_served_[s] = 0;
  return evicted;
}

int NeuralCpu::set_read_lease_bytes(int addr, const std::uint8_t* values, long n) {
  addr &= 0xFFFF;
  const auto s = static_cast<size_t>(lease_claim(addr));
  const int evicted = lease_addr_[s] == addr ? -1 : lease_addr_[s];
  lease_addr_[s] = addr;
  lease_n_[s] = n < 8 ? static_cast<int>(n) : 8;
  lease_i_[s] = 0;
  lease_served_[s] = 0;
  for (int i = 0; i < lease_n_[s]; ++i)
    lease_values_[s][static_cast<size_t>(i)] = values[i];
  if (lease_n_[s] == 0)
    lease_addr_[s] = -1;
  return evicted;
}

void NeuralCpu::clear_read_lease() {
  for (size_t s = 0; s < lease_capacity; ++s) {
    lease_addr_[s] = -1;
    lease_n_[s] = lease_i_[s] = 0;
    lease_served_[s] = 0;
  }
}

// Targeted invalidation (a write only disturbs its own device's registers);
// the bus flushes served counts first.
void NeuralCpu::clear_read_lease_addr(int addr) {
  const int s = lease_slot(addr & 0xFFFF);
  if (s >= 0) {
    const auto k = static_cast<size_t>(s);
    lease_addr_[k] = -1;
    lease_n_[k] = lease_i_[k] = 0;
    lease_served_[k] = 0;
  }
}

long NeuralCpu::take_lease_served(int addr) {
  const int s = lease_slot(addr & 0xFFFF);
  if (s < 0)
    return 0;
  return std::exchange(lease_served_[static_cast<size_t>(s)], 0);
}

} // namespace transformer_nes::nes
