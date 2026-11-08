#pragma once

// 2A03 neural state and sample circuits; the host schedules and transports effects.

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::function;
using std::int64_t;
using std::uint8_t;
using std::vector;

constexpr double cpu_frequency_hz = 1'789'772.727272727; // NTSC 2A03

// State of a 4-bit envelope, advanced by the joint modulation token.
struct Envelope {
  bool start = false;
  bool loop = false; // == length-halt flag
  bool constant = false;
  int period = 0; // also the constant volume
  int divider = 0;
  int decay = 0;
};

struct Pulse {
  // Programming.
  uint8_t duty = 0;
  int timer = 0; // 11-bit period
  int length = 0;
  bool enabled = false;
  Envelope env;
  // Sweep.
  bool sweep_enable = false;
  int sweep_period = 0;
  bool sweep_negate = false;
  int sweep_shift = 0;
  bool sweep_reload = false;
  int sweep_divider = 0;
  // Runtime.
  int timer_counter = 0;
  uint8_t seq_pos = 0;
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
};

struct Noise {
  int length = 0;
  bool enabled = false;
  Envelope env;
  bool mode = false; // short-period feedback tap
  int period = 0;    // CPU cycles
  int timer_counter = 0;
  array<int, 15> lfsr{1}; // neural bit lanes, least significant first
};

// Fixed int32 token state in audio.py:DMC_STATE order.
struct Dmc {
  int irq_enabled = 0;
  int loop = 0;
  int irq = 0;
  int period = 428;
  int output_level = 0;
  int sample_address = 0xC000;
  int sample_length = 1;
  int current_address = 0xC000;
  int bytes_remaining = 0;
  int sample_buffer = 0;
  int buffer_empty = 1;
  int shift_register = 0;
  int bits_remaining = 8;
  int silence = 1;
};

class Apu {
public:
  using MemoryReader = function<int(int)>;

  explicit Apu(double rate = 44'100.0) {
    if (rate <= 0.0)
      throw std::invalid_argument("APU sample rate must be positive");
    sample_rate = rate;
    cycles_per_sample = cpu_frequency_hz / rate;
  }

  enum class FrameEvent { Clock, Write, Read, Observe };
  enum class DmcEvent { Write, Fetch, Clock, Observe };
  static constexpr int channel_input_count = 52, channel_output_count = 21;
  static constexpr int sample_input_count = 13, sample_parameter_count = 4;
  // The APU chips' model ids (apu2a03/audio.py:chip, mixer_chip).
  struct Engine {
    void* ctx = nullptr;
    void (*step)(void*, int, const int*, int*) = nullptr;
    int control = 0, frame = 0, program = 0;
    array<int, 3> dmc{}; // full (writes, observation), fetch, clock
    array<int, 2> oscillators{}; // [one clock]
    array<array<int, 2>, 2> noise{}; // [one noise step][noise mode]
    int span = 0; // the longest span one oscillator forward covers
    void* sample_ctx = nullptr;
    void (*sample_step)(void*, int, const float*, float*) = nullptr;
    int sample_model = 0;
    int parameter_model = 0;
  };
  void set_engine(Engine engine);

  int read(int address);
  // Returns the program token's DMC scheduling effect.
  bool write(int address, int value);
  // Advance state to a stamped CPU cycle before applying an MMIO write.
  void advance_to(int64_t cpu_cycle);
  void take_samples(vector<float>& out);
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
  int seq_step = 0;
  int frame_deadline = 0;
  bool apu_tick = false; // toggles each CPU cycle -> APU-rate clock

  // Output.
  double sample_rate = 0.0;
  double cycles_per_sample = 0.0;
  double sample_accum = 0.0;
  float hp_prev_in = 0.0f, hp_prev_out = 0.0f; // DC-blocker state
  int dma_stall = 0;
  int64_t clocked_cycles = 0;
  vector<float> pending_samples;
  // CPU memory as the DMC's sample DMA reads it (empty: no fetches).
  MemoryReader memory;

private:
  struct AccessResult { int read_value; bool dmc_moved; };
  AccessResult access(int port, int value, bool writing);
  void load_program_state(const int* fields);
  void dmc_event(DmcEvent event, int port = 0, int value = 0);
  // DMC scheduling effects; digital reader/channel state stays in tokens.
  int dmc_timer_ = 0, dmc_tail_ = 0;
  bool dmc_fetch_due_ = false, dmc_dma_active_ = false;
  void fill_dmc_buffer();
  void clock_dmc();
  Engine engine_;
  array<int, channel_input_count> channel_inputs(int event) const;
  void channel_event(int event);
  void frame_event(FrameEvent event, int value = 0);
  void clock_frame_sequencer();
  float sample();
  array<float, sample_parameter_count> sample_parameters_{};
  bool sample_parameters_dirty_ = true;
  void advance_oscillators(int cycles);
};

} // namespace transformer_nes::nes
