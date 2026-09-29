#pragma once

// Native NES APU (2A03 audio): registers $4000-$4017 behind the MMIO
// bus, the five channels' state, a frame sequencer, and continuous samples.

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::function;
using std::int64_t;
using std::size_t;
using std::uint16_t;
using std::uint8_t;
using std::vector;

constexpr double cpu_frequency_hz = 1'789'772.727272727; // NTSC 2A03

// Length counter load values indexed by the 5-bit field in $4003/$400B/etc.
inline constexpr array<uint8_t, 32> length_table = {
    10, 254, 20, 2,  40, 4,  80, 6,  160, 8,  60, 10, 14, 12, 26, 14,
    12, 16,  24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
};
// Pulse duty sequences (8 steps each), high = channel on.
inline constexpr array<array<uint8_t, 8>, 4> duty_sequences = {{
    {0, 1, 0, 0, 0, 0, 0, 0}, // 12.5%
    {0, 1, 1, 0, 0, 0, 0, 0}, // 25%
    {0, 1, 1, 1, 1, 0, 0, 0}, // 50%
    {1, 0, 0, 1, 1, 1, 1, 1}, // 25% negated
}};
// Triangle 32-step sequence (0..15..0).
inline constexpr array<uint8_t, 32> triangle_sequence = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,  4,  3,  2,  1,  0,
    0,  1,  2,  3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};
// Noise timer periods (CPU cycles) indexed by the 4-bit field, NTSC.
inline constexpr array<uint16_t, 16> noise_periods = {
    4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
};
inline constexpr array<uint16_t, 16> dmc_periods = {
    428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 85, 72, 54,
};
inline constexpr array<int, 4> four_step_cycles = {7457, 14913, 22371, 29829};
inline constexpr array<int, 4> five_step_cycles = {7457, 14913, 22371, 37281};

// How many times a down-counter at `counter` with reload `period` expires
// over n clocks (an expiry when it is clocked at 0), leaving `counter` as
// n single clocks would.
inline int timer_expiries(int& counter, int period, int n) {
  if (n <= counter) {
    counter -= n;
    return 0;
  }
  const int rest = n - counter - 1, span = period + 1;
  counter = period - rest % span;
  return 1 + rest / span;
}

// A 4-bit-output envelope generator shared by pulse and noise.
struct Envelope {
  bool start = false;
  bool loop = false; // == length-halt flag
  bool constant = false;
  int period = 0; // also the constant volume
  int divider = 0;
  int decay = 0;

  void clock() {
    if (start) {
      start = false;
      decay = 15;
      divider = period;
    } else if (divider-- == 0) {
      divider = period;
      if (decay > 0)
        decay--;
      else if (loop)
        decay = 15;
    }
  }
  int volume() const { return constant ? period : decay; }
};

struct Pulse {
  // Programming.
  uint8_t duty = 0;
  int timer = 0; // 11-bit period
  int length = 0;
  bool length_halt = false;
  bool enabled = false;
  Envelope env;
  // Sweep.
  bool sweep_enable = false;
  int sweep_period = 0;
  bool sweep_negate = false;
  int sweep_shift = 0;
  bool sweep_reload = false;
  int sweep_divider = 0;
  bool ones_complement = false; // pulse 1 negates with one's complement
  // Runtime.
  int timer_counter = 0;
  uint8_t seq_pos = 0;

  int target_period() const {
    const int change = timer >> sweep_shift;
    if (sweep_negate)
      return timer - change - (ones_complement ? 1 : 0);
    return timer + change;
  }
  bool muted() const { return timer < 8 || target_period() > 0x7FF; }

  void clock_timer() { // called at APU rate (every 2 CPU cycles)
    if (timer_counter-- == 0) {
      timer_counter = timer;
      seq_pos = (seq_pos + 1) & 7;
    }
  }
  void advance(int ticks) { // `ticks` clock_timer calls
    seq_pos = static_cast<uint8_t>((seq_pos + timer_expiries(timer_counter, timer, ticks)) & 7);
  }
  void clock_length() {
    if (length > 0 && !length_halt)
      length--;
  }
  void clock_sweep() {
    if (sweep_divider == 0 && sweep_enable && sweep_shift > 0 && !muted()) {
      const int t = target_period();
      if (t >= 0 && t <= 0x7FF)
        timer = t;
    }
    if (sweep_divider == 0 || sweep_reload) {
      sweep_divider = sweep_period;
      sweep_reload = false;
    } else {
      sweep_divider--;
    }
  }
  int output() const {
    if (!enabled || length == 0 || muted() || !duty_sequences[duty][seq_pos])
      return 0;
    return env.volume();
  }
};

struct Triangle {
  int timer = 0;
  int length = 0;
  bool length_halt = false; // also the linear-counter control flag
  bool enabled = false;
  int linear_reload_value = 0;
  bool linear_reload = false;
  int linear_counter = 0;
  int timer_counter = 0;
  uint8_t seq_pos = 0;

  void clock_timer() { // called at CPU rate
    if (timer_counter-- == 0) {
      timer_counter = timer;
      if (length > 0 && linear_counter > 0)
        seq_pos = (seq_pos + 1) & 31;
    }
  }
  void advance(int cycles) { // `cycles` clock_timer calls
    const int steps = timer_expiries(timer_counter, timer, cycles);
    if (length > 0 && linear_counter > 0)
      seq_pos = static_cast<uint8_t>((seq_pos + steps) & 31);
  }
  void clock_length() {
    if (length > 0 && !length_halt)
      length--;
  }
  void clock_linear() {
    if (linear_reload)
      linear_counter = linear_reload_value;
    else if (linear_counter > 0)
      linear_counter--;
    if (!length_halt)
      linear_reload = false;
  }
  int output() const {
    if (!enabled || length == 0 || linear_counter == 0)
      return 0;
    // Very high frequencies (timer < 2) produce a ~DC output on hardware;
    // silence them to avoid a full-volume click.
    if (timer < 2)
      return 0;
    return triangle_sequence[seq_pos];
  }
};

struct Noise {
  int length = 0;
  bool length_halt = false;
  bool enabled = false;
  Envelope env;
  bool mode = false; // short-period feedback tap
  int period = 0;    // CPU cycles
  int timer_counter = 0;
  uint16_t lfsr = 1;

  void clock_timer() { // called at APU rate
    if (timer_counter-- == 0) {
      timer_counter = period;
      step_lfsr();
    }
  }
  void advance(int ticks) { // `ticks` clock_timer calls
    for (int k = timer_expiries(timer_counter, period, ticks); k > 0; --k)
      step_lfsr();
  }
  void step_lfsr() {
    const int bit = mode ? 6 : 1;
    const uint16_t fb = (lfsr & 1) ^ ((lfsr >> bit) & 1);
    lfsr = static_cast<uint16_t>((lfsr >> 1) | (fb << 14));
  }
  void clock_length() {
    if (length > 0 && !length_halt)
      length--;
  }
  int output() const {
    if (!enabled || length == 0 || (lfsr & 1))
      return 0;
    return env.volume();
  }
};

struct Dmc {
  bool irq_enabled = false;
  bool loop = false;
  bool irq = false;
  int period = dmc_periods.front();
  int timer_counter = dmc_periods.front();
  int output_level = 0;
  uint16_t sample_address = 0xC000;
  int sample_length = 1;
  uint16_t current_address = 0xC000;
  int bytes_remaining = 0;
  int sample_buffer = 0;
  bool buffer_empty = true;
  int shift_register = 0;
  int bits_remaining = 8;
  bool silence = true;

  void restart() {
    current_address = sample_address;
    bytes_remaining = sample_length;
  }
};

class Apu {
public:
  using MemoryReader = function<int(int)>;

  explicit Apu(double rate = 44'100.0) {
    if (rate <= 0.0)
      throw std::invalid_argument("APU sample rate must be positive");
    pulse[0].ones_complement = true;
    sample_rate = rate;
    cycles_per_sample = cpu_frequency_hz / rate;
  }

  int read(int address);
  void write(int address, int value);
  // Advance state to a stamped CPU cycle before applying an MMIO write.
  void advance_to(int64_t cpu_cycle);
  [[nodiscard]] vector<float> take_samples();
  [[nodiscard]] int take_dma_stall() {
    const int stall = dma_stall;
    dma_stall = 0;
    return stall;
  }
  [[nodiscard]] int64_t cycles_until_dmc_dma() const;
  [[nodiscard]] bool irq_pending() const { return frame_irq || dmc.irq; }

  array<Pulse, 2> pulse{};
  Triangle triangle;
  Noise noise;
  Dmc dmc;

  // Frame sequencer.
  bool five_step = false;
  bool irq_inhibit = false;
  bool frame_irq = false;
  int seq_cycle = 0; // CPU cycles into the current sequence
  size_t seq_step = 0;
  bool apu_tick = false; // toggles each CPU cycle -> APU-rate clock

  // Output.
  double sample_rate = 0.0;
  double cycles_per_sample = 0.0;
  double sample_accum = 0.0;
  double hp_prev_in = 0.0, hp_prev_out = 0.0; // DC-blocker state
  int dma_stall = 0;
  int64_t clocked_cycles = 0;
  vector<float> pending_samples;
  // CPU memory as the DMC's sample DMA reads it (empty: no fetches).
  MemoryReader memory;

private:
  void fill_dmc_buffer();
  void clock_dmc();
  void clock_quarter();
  void clock_half();
  void clock_frame_sequencer();
  double mix_level() const;
  float sample();
  void clock_cycle();
};

} // namespace transformer_nes::nes
