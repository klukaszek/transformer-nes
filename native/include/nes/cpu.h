#pragma once

// The 6502 run loop over the model's compiled CPU chip (cpu6502/chip.py).
//
// The weights compute every instruction; this host keeps the architectural
// registers between steps, embeds them as a step's inputs and commits its
// outputs, and arbitrates what the weights cannot see: device reads served
// by leases, watched writes, interrupt lines and run budgets. Which circuit
// runs comes from tables the model's Python side computes (CpuTables).

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace transformer_nes::nes {

// Which compiled circuit runs, by situation (cpu6502/chip.py:dispatch).
struct CpuTables {
  static constexpr int none = -1;
  int union_model = 0;                   // any byte, either interrupt
  std::array<int, 256> exec{};           // an opcode's execution, or none
  std::array<bool, 256> exec_after_read{}; // runs only once its device read is served
  std::array<int, 256> request{};        // an opcode's device-read request, or none
  int nmi = none, irq = none;
  struct Loop {
    int reg, delta, model;
  };
  std::vector<Loop> loops;               // counter recurrences by register and step
  // A fixed window of instructions run as one forward.
  struct Shape {
    int model, length, steps, prefix, spin;
    std::vector<std::pair<int, int>> checks; // (offset, opcode) the window must hold
    bool self_target;                    // a JMP to itself
    std::array<int, 5> loop;             // [kind, closure offset, closure byte, register, delta]
  };
  std::vector<Shape> shapes;
  // Trace tables (cpu6502/engine.py trace_*_table): addressing mode,
  // instruction length and the data a trace row reports, by opcode.
  std::array<std::uint8_t, 256> trace_mode{}, trace_length{}, trace_data{};
  // Per model id: whether its circuit reads the interrupt lines (and so
  // takes a pending interrupt it is shown).
  std::vector<std::uint8_t> reads_interrupts;
};

class NeuralCpu {
public:
  using Step = void (*)(void*, int, const int*, int*);
  static constexpr int lease_capacity = 4;
  // The chip's token layouts (cpu6502/chip.py INPUTS, STEP_OUTPUTS, ...).
  static constexpr int inputs = 37, step_outputs = 25, union_outputs = 24, request_outputs = 4;

  NeuralCpu(void* engine, Step step, std::uint8_t* memory, CpuTables tables, bool mirror_cpu_ram);
  // Run on another engine of the same chip (an observing build), its
  // memory taking this one's contents.
  void rebind(void* engine, Step step, std::uint8_t* memory);

  struct Run {
    const char* reason;
    int read_addr;
    long read_cycles, cycles, steps, forwards;
  };
  Run run(long max_steps, bool stop_on_trap, long max_cycles, bool trace);

  void reset();
  void set_state(int pc, int a, int x, int y, int s, int p, int ir, int oplo, int ophi, int ea,
                 int data, bool halted);
  struct State {
    int pc, a, x, y, s, p, ir, oplo, ophi, ea, data;
    bool halted;
    long steps, cycles;
  };
  [[nodiscard]] State state() const;

  void watch_read(int lo, int hi) { rwatch_.add(lo, hi); }
  void watch_write(int lo, int hi) { wwatch_.add(lo, hi); }
  void clear_watches() { rwatch_.clear(); wwatch_.clear(); }
  // Leases: a device read answered without pausing, one repeated value or a
  // finite reply queue. Setters return the evicted address, or -1.
  int set_read_lease(int addr, int value);
  int set_read_lease_bytes(int addr, const std::uint8_t* values, long n);
  void clear_read_lease();
  void clear_read_lease_addr(int addr);
  long take_lease_served(int addr);
  void set_nmi() { nmi_pending_ = true; }
  void set_irq() { irq_pending_ = true; }
  void clear_irq() { irq_pending_ = false; }

  [[nodiscard]] int read_ram(int addr) const { return ram_[addr & 0xFFFF]; }
  [[nodiscard]] const std::uint8_t* ram() const { return ram_; }
  // A CPU-visible write (mirrored through internal RAM), or a mapped byte.
  void inject_write(int addr, int value);
  void map_bytes(int base, const std::uint8_t* data, long n);
  void seed_image(const std::uint8_t* image); // 64 KiB
  void add_window(int shape, int head);
  void clear_windows();

  // The last run's per-step trace and writes, and watched writes hit.
  struct Trace {
    std::vector<int> pc, a, x, y, s, p, ir, oplo, ophi, ea, data, halted;
    std::vector<long> w_row;
    std::vector<int> w_addr, w_value, w_original;
    void clear();
  };
  [[nodiscard]] const Trace& trace() const { return trace_; }
  struct Hits {
    std::vector<int> addr, value, original; // original: -1, or an RMW operand
  };
  [[nodiscard]] const Hits& hits() const { return hits_; }

  // The X-ray: instructions retired per path (shapes, interrupts, opcodes,
  // the union) and the dispatches that retired them, and per opcode.
  std::array<long, 4> xray_paths{}, xray_dispatches{};
  std::array<long, 256> xray_opcodes{};
  // The union's inputs for the current state, as the X-ray traces it.
  std::array<int, inputs> union_inputs();

private:
  struct ReadRequest {
    int address = -1, cycles = 0, oplo = 0, ophi = 0;
  };
  struct Watches {
    std::array<std::uint64_t, 1024> words{};
    bool any = false;
    [[nodiscard]] bool empty() const { return !any; }
    [[nodiscard]] bool contains(int addr) const {
      return static_cast<unsigned>(addr) < 65536 && (words[addr >> 6] & (std::uint64_t{1} << (addr & 63)));
    }
    void add(int lo, int hi);
    void clear() {
      if (any) {
        words.fill(0);
        any = false;
      }
    }
  };
  enum { LOOP_NONE, LOOP_POLL, LOOP_COUNTER };

  // The current step's inputs: the state's own fields, and the context
  // that varies per step (PC, effective address, the read, loops, lines).
  int* embed(int model, bool consume = true);
  void forward(int model, const int* in, int* out) { step_(engine_, model, in, out); }
  void step(int model, bool stage_writes);
  bool execution_step(int opcode, bool read_serviced);
  bool interrupt_step(bool nmi);
  ReadRequest request_read();
  void union_step(long long* o);
  void counter_recurrence(int reg, int delta, int limit, int cycles);
  int shape_dispatch(long i, long max_steps, long cycles_this, long max_cycles, bool stop_on_trap);
  int run_shape(const CpuTables::Shape& shape, long i, long max_steps, long cycles_this,
                long max_cycles);
  int commit_writes(bool trace);
  void write(int addr, int value);
  int lease_slot(int addr) const;
  int lease_claim(int addr);
  [[nodiscard]] bool irq_ready() const { return irq_pending_ && !nmi_pending_ && !in_[in_p + 2]; }
  [[nodiscard]] int p() const;
  void set_p(int p);
  void set_a(int a);
  int trace_ea(int opcode, int pc, int xr, int yr) const;
  int trace_data(int opcode, int pre_a, int pre_x, int pre_y, int pre_p, int operand) const;
  void trace_row(int pc, int a, int x, int y, int s, int p, int ir, int oplo, int ophi, int ea,
                 int data, bool halted);

  void* engine_;
  Step step_;
  std::uint8_t* ram_; // the chip's memory: what its circuits read
  CpuTables tables_;
  bool mirror_cpu_ram_;

  // The state as the circuits read it: registers, P's flags and A's bits
  // live in their input fields, so a step's outputs commit into the next
  // step's inputs (cpu6502/chip.py INPUTS).
  static constexpr int in_a_field = 3, in_x_field = 4, in_y_field = 5, in_s_field = 6, in_ir_field = 7,
                       in_oplo_field = 8, in_ophi_field = 9, in_data_field = 12, in_halted_field = 13,
                       in_p = 21, in_a_bits = 29;
  std::array<int, inputs> in_{};
  int& a_ = in_[in_a_field];
  int& x_ = in_[in_x_field];
  int& y_ = in_[in_y_field];
  int& s_ = in_[in_s_field];
  int& ir_ = in_[in_ir_field];
  int& oplo_ = in_[in_oplo_field];
  int& ophi_ = in_[in_ophi_field];
  int& data_ = in_[in_data_field];
  int& halted_ = in_[in_halted_field];
  int pc_ = 0, ea_ = 0;
  bool nmi_pending_ = false, irq_pending_ = false, resume_read_skip_ = false;
  ReadRequest pending_read_;
  int read_ir_ = 0;
  long seq_cycles_ = 0;
  int loop_limit_ = 0, loop_cycles_ = 0, loop_iterations_ = 0;
  int n_writes_ = 0;
  std::array<int, 3> write_addr_{}, write_value_{}, write_original_{};
  std::array<int, lease_capacity> lease_addr_{}, lease_value_{}, lease_n_{}, lease_i_{};
  std::array<std::array<int, 8>, lease_capacity> lease_values_{};
  std::array<long, lease_capacity> lease_served_{};
  long step_index_ = 0, cycles_total_ = 0;
  Watches rwatch_, wwatch_;
  std::vector<std::uint8_t> shape_of_pc_;
  Trace trace_;
  Hits hits_;
};

} // namespace transformer_nes::nes
