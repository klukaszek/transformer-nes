// Native NES APU (2A03 audio).  Registers $4000-$4017 behind the MMIO
// bus, the five channels' state, a frame sequencer, and a per-frame
// sample generator that advances the channels by a frame's CPU cycles
// and mixes them with the 2A03's nonlinear DAC.  Plain fixed silicon --
// nothing here is model-generated -- compiled once and cached by content
// digest, like the PPU.
//
// Frame-level approximation (documented, revisit when a game needs it):
// register writes take effect at frame granularity and the frame's
// samples are generated in one batch at frame end, so the vast-majority
// case of once-per-frame (NMI) music updates is exact while a rapid
// intra-frame sweep resolves at frame granularity. DMC memory fetches use
// a callback into the console's current CPU image; their four-cycle DMA
// stalls are reported to the console for clock accounting.

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "nes/apu.h"

namespace transformer_nes::nes {

using std::array;
using std::size_t;
using std::uint16_t;
using std::uint8_t;

namespace {

constexpr double cpu_frequency_hz = 1'789'772.727272727; // NTSC 2A03

// Length counter load values indexed by the 5-bit field in $4003/$400B/etc.
constexpr array<uint8_t, 32> length_table = {
    10, 254, 20, 2,  40, 4,  80, 6,  160, 8,  60, 10, 14, 12, 26, 14,
    12, 16,  24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
};
// Pulse duty sequences (8 steps each), high = channel on.
constexpr array<array<uint8_t, 8>, 4> duty_sequences = {{
    {0, 1, 0, 0, 0, 0, 0, 0}, // 12.5%
    {0, 1, 1, 0, 0, 0, 0, 0}, // 25%
    {0, 1, 1, 1, 1, 0, 0, 0}, // 50%
    {1, 0, 0, 1, 1, 1, 1, 1}, // 25% negated
}};
// Triangle 32-step sequence (0..15..0).
constexpr array<uint8_t, 32> triangle_sequence = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,  4,  3,  2,  1,  0,
    0,  1,  2,  3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};
// Noise timer periods (CPU cycles) indexed by the 4-bit field, NTSC.
constexpr array<uint16_t, 16> noise_periods = {
    4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
};
constexpr array<uint16_t, 16> dmc_periods = {
    428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 85, 72, 54,
};
constexpr array<int, 4> four_step_cycles = {7457, 14913, 22371, 29829};
constexpr array<int, 4> five_step_cycles = {7457, 14913, 22371, 37281};

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
      const int bit = mode ? 6 : 1;
      const uint16_t fb = (lfsr & 1) ^ ((lfsr >> bit) & 1);
      lfsr = static_cast<uint16_t>((lfsr >> 1) | (fb << 14));
    }
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

} // namespace

struct Apu::Impl {
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
  double sample_rate;
  double cycles_per_sample;
  double sample_accum = 0.0;
  double hp_prev_in = 0.0, hp_prev_out = 0.0; // DC-blocker state
  function<int(int)> memory_reader;
  int dma_stall = 0;

  explicit Impl(double rate) {
    if (rate <= 0.0)
      throw std::invalid_argument("APU sample rate must be positive");
    pulse[0].ones_complement = true;
    sample_rate = rate;
    cycles_per_sample = cpu_frequency_hz / rate;
    noise.lfsr = 1;
  }

  // ── Device protocol ($4000-$4017) ───────────────────────────────

  int read(int addr) {
    if ((addr & 0xFFFF) == 0x4015) {
      int v = 0;
      if (pulse[0].length > 0)
        v |= 0x01;
      if (pulse[1].length > 0)
        v |= 0x02;
      if (triangle.length > 0)
        v |= 0x04;
      if (noise.length > 0)
        v |= 0x08;
      if (dmc.bytes_remaining > 0)
        v |= 0x10;
      if (frame_irq)
        v |= 0x40;
      if (dmc.irq)
        v |= 0x80;
      frame_irq = false; // reading $4015 clears the frame IRQ
      return v;
    }
    return 0x40; // open bus (also serves controller-2 reads at $4017)
  }

  void write(int addr, int value) {
    value &= 0xFF;
    switch (addr & 0xFFFF) {
    case 0x4000:
    case 0x4004: {
      Pulse& p = pulse[(addr >> 2) & 1];
      p.duty = value >> 6;
      p.length_halt = value & 0x20;
      p.env.loop = value & 0x20;
      p.env.constant = value & 0x10;
      p.env.period = value & 0x0F;
      break;
    }
    case 0x4001:
    case 0x4005: {
      Pulse& p = pulse[(addr >> 2) & 1];
      p.sweep_enable = value & 0x80;
      p.sweep_period = (value >> 4) & 7;
      p.sweep_negate = value & 0x08;
      p.sweep_shift = value & 7;
      p.sweep_reload = true;
      break;
    }
    case 0x4002:
    case 0x4006: {
      Pulse& p = pulse[(addr >> 2) & 1];
      p.timer = (p.timer & 0x700) | value;
      break;
    }
    case 0x4003:
    case 0x4007: {
      Pulse& p = pulse[(addr >> 2) & 1];
      p.timer = (p.timer & 0xFF) | ((value & 7) << 8);
      if (p.enabled)
        p.length = length_table[value >> 3];
      p.seq_pos = 0;
      p.env.start = true;
      break;
    }
    case 0x4008:
      triangle.length_halt = value & 0x80;
      triangle.linear_reload_value = value & 0x7F;
      break;
    case 0x400A:
      triangle.timer = (triangle.timer & 0x700) | value;
      break;
    case 0x400B:
      triangle.timer = (triangle.timer & 0xFF) | ((value & 7) << 8);
      if (triangle.enabled)
        triangle.length = length_table[value >> 3];
      triangle.linear_reload = true;
      break;
    case 0x400C:
      noise.length_halt = value & 0x20;
      noise.env.loop = value & 0x20;
      noise.env.constant = value & 0x10;
      noise.env.period = value & 0x0F;
      break;
    case 0x400E:
      noise.mode = value & 0x80;
      noise.period = noise_periods[value & 0x0F];
      break;
    case 0x400F:
      if (noise.enabled)
        noise.length = length_table[value >> 3];
      noise.env.start = true;
      break;
    case 0x4010:
      dmc.irq_enabled = value & 0x80;
      dmc.loop = value & 0x40;
      dmc.period = dmc_periods[value & 0x0F];
      if (!dmc.irq_enabled)
        dmc.irq = false;
      break;
    case 0x4011:
      dmc.output_level = value & 0x7F;
      break;
    case 0x4012:
      dmc.sample_address = static_cast<uint16_t>(0xC000 | (value << 6));
      break;
    case 0x4013:
      dmc.sample_length = (value << 4) | 1;
      break;
    case 0x4015:
      pulse[0].enabled = value & 0x01;
      pulse[1].enabled = value & 0x02;
      triangle.enabled = value & 0x04;
      noise.enabled = value & 0x08;
      if (!pulse[0].enabled)
        pulse[0].length = 0;
      if (!pulse[1].enabled)
        pulse[1].length = 0;
      if (!triangle.enabled)
        triangle.length = 0;
      if (!noise.enabled)
        noise.length = 0;
      if (!(value & 0x10))
        dmc.bytes_remaining = 0;
      else if (dmc.bytes_remaining == 0)
        dmc.restart();
      dmc.irq = false;
      break;
    case 0x4017:
      five_step = value & 0x80;
      irq_inhibit = value & 0x40;
      if (irq_inhibit)
        frame_irq = false;
      seq_cycle = 0;
      seq_step = 0;
      if (five_step) { // 5-step immediately clocks env + length
        clock_quarter();
        clock_half();
      }
      break;
    default:
      break; // $4009/$400D unused
    }
  }

  bool irq_pending() const { return frame_irq || dmc.irq; }

  void fill_dmc_buffer() {
    if (!dmc.buffer_empty || dmc.bytes_remaining == 0 || !memory_reader)
      return;
    dmc.sample_buffer = memory_reader(dmc.current_address) & 0xFF;
    dmc.buffer_empty = false;
    dma_stall += 4;
    dmc.current_address =
        dmc.current_address == 0xFFFF ? 0x8000 : uint16_t(dmc.current_address + 1);
    if (--dmc.bytes_remaining == 0) {
      if (dmc.loop)
        dmc.restart();
      else if (dmc.irq_enabled)
        dmc.irq = true;
    }
  }

  void clock_dmc() {
    fill_dmc_buffer();
    if (dmc.timer_counter-- > 0)
      return;
    dmc.timer_counter = dmc.period - 1;
    if (!dmc.silence) {
      if (dmc.shift_register & 1) {
        if (dmc.output_level <= 125)
          dmc.output_level += 2;
      } else if (dmc.output_level >= 2) {
        dmc.output_level -= 2;
      }
    }
    dmc.shift_register >>= 1;
    if (--dmc.bits_remaining == 0) {
      dmc.bits_remaining = 8;
      if (dmc.buffer_empty) {
        dmc.silence = true;
      } else {
        dmc.silence = false;
        dmc.shift_register = dmc.sample_buffer;
        dmc.buffer_empty = true;
      }
    }
  }

  // ── Frame sequencer ─────────────────────────────────────────────

  void clock_quarter() { // envelopes + triangle linear counter
    pulse[0].env.clock();
    pulse[1].env.clock();
    noise.env.clock();
    triangle.clock_linear();
  }
  void clock_half() { // length counters + sweeps
    pulse[0].clock_length();
    pulse[1].clock_length();
    triangle.clock_length();
    noise.clock_length();
    pulse[0].clock_sweep();
    pulse[1].clock_sweep();
  }

  // Advance the sequencer by one CPU cycle, firing its steps at the NTSC
  // trigger cycles (expressed in CPU cycles = 2x the APU-cycle values).
  void clock_frame_sequencer() {
    const auto& events = five_step ? five_step_cycles : four_step_cycles;
    if (++seq_cycle != events[seq_step])
      return;
    clock_quarter();
    if (seq_step == 1 || seq_step == 3)
      clock_half();
    if (seq_step != 3) {
      ++seq_step;
      return;
    }
    if (!five_step && !irq_inhibit)
      frame_irq = true;
    seq_cycle = 0;
    seq_step = 0;
  }

  // ── Mixing (2A03 nonlinear DAC + DC blocker) ────────────────────

  // Raw mixed level in ~[0, 1] from the two nonlinear DAC groups.
  double mix_level() const {
    const int p = pulse[0].output() + pulse[1].output();
    const int t = triangle.output();
    const int n = noise.output();
    double pulse_out = 0.0, tnd_out = 0.0;
    if (p)
      pulse_out = 95.88 / (8128.0 / p + 100.0);
    const double tnd = t / 8227.0 + n / 12241.0 + dmc.output_level / 22638.0;
    if (tnd > 0.0)
      tnd_out = 159.79 / (1.0 / tnd + 100.0);
    return pulse_out + tnd_out;
  }

  // One-pole high-pass (the 2A03's own DC blocking): removes the DAC's
  // large DC offset so the signal is centered, then scale to line level.
  float sample() {
    const double in = mix_level();
    const double out = in - hp_prev_in + 0.9995 * hp_prev_out;
    hp_prev_in = in;
    hp_prev_out = out;
    double s = out * 2.0;
    if (s > 1.0)
      s = 1.0;
    else if (s < -1.0)
      s = -1.0;
    return static_cast<float>(std::clamp(s, -1.0, 1.0));
  }

  // Advance every channel by one CPU cycle (triangle each cycle; the
  // others at APU rate, every other cycle).
  void clock_cycle() {
    triangle.clock_timer();
    clock_dmc();
    apu_tick = !apu_tick;
    if (apu_tick) {
      pulse[0].clock_timer();
      pulse[1].clock_timer();
      noise.clock_timer();
    }
    clock_frame_sequencer();
  }

  // Run a frame's worth of CPU cycles, emitting one sample every
  // cycles_per_sample cycles; returns the generated mono samples.
  vector<float> end_frame(int cpu_cycles, function<int(int)> reader = {}) {
    memory_reader = std::move(reader);
    struct ResetReader {
      function<int(int)>& reader;
      ~ResetReader() { reader = {}; }
    } reset_reader{memory_reader};
    // sample_accum is kept in [0, cycles_per_sample), so this is the
    // exact number the loop below will emit.  Write straight into the
    // exact result size instead of growing the vector during every frame.
    const auto sample_count = static_cast<size_t>((sample_accum + cpu_cycles) / cycles_per_sample);
    vector<float> out(sample_count);
    size_t written = 0;
    for (int i = 0; i < cpu_cycles; ++i) {
      clock_cycle();
      if (++sample_accum >= cycles_per_sample) {
        sample_accum -= cycles_per_sample;
        out[written++] = sample();
      }
    }
    return out;
  }

  int take_dma_stall() {
    const int stall = dma_stall;
    dma_stall = 0;
    return stall;
  }
};

Apu::Apu(double sample_rate) : impl_(std::make_unique<Impl>(sample_rate)) {}
Apu::~Apu() = default;
Apu::Apu(Apu&&) noexcept = default;
Apu& Apu::operator=(Apu&&) noexcept = default;

int Apu::read(int address) { return impl_->read(address); }

void Apu::write(int address, int value) { impl_->write(address, value); }

vector<float> Apu::end_frame(int cpu_cycles, MemoryReader reader) {
  return impl_->end_frame(cpu_cycles, std::move(reader));
}

int Apu::take_dma_stall() { return impl_->take_dma_stall(); }

bool Apu::irq_pending() const { return impl_->irq_pending(); }

} // namespace transformer_nes::nes
