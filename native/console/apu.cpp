// NES APU (2A03): neural frame/channel control and digital levels,
// sample generation, and DMC memory reads on the CPU-clock timeline.

#include <algorithm>
#include <bit>
#include <cmath>

#include "nes/apu.h"

namespace transformer_nes::nes {

// Device protocol ($4000-$4017)

int Apu::read(int addr) {
  return access((addr & 0xFFFF) - 0x4000, 0, false).read_value;
}

bool Apu::write(int addr, int value) {
  return access((addr & 0xFFFF) - 0x4000, value & 0xFF, true).dmc_moved;
}

Apu::AccessResult Apu::access(int port, int value, bool writing) {
  if (!engine_.ctx)
    throw std::logic_error("APU transformer engine is required");
  const auto channels = channel_inputs(0);
  int in[59];
  int* field = std::copy(channels.begin() + 1, channels.end(), in);
  for (const int data : {int(noise.mode), noise.period, port, value, int(writing),
                         dmc.bytes_remaining, int(dmc.irq), int(frame_irq)})
    *field++ = data;
  int out[57];
  engine_.step(engine_.ctx, engine_.program, in, out);
  load_program_state(out);
  sample_parameters_dirty_ |= writing;
  if (out[54])
    frame_event(FrameEvent::Write, value);
  if (out[55])
    frame_event(FrameEvent::Read);
  if (out[56])
    dmc_event(DmcEvent::Write, port, value);
  return {out[53], out[56] != 0};
}

void Apu::load_program_state(const int* field) {
  auto envelope = [&field](Envelope& e) {
    e.start = *field++ != 0;
    e.loop = *field++ != 0;
    e.constant = *field++ != 0;
    e.period = *field++;
    e.divider = *field++;
    e.decay = *field++;
  };
  for (auto& p : pulse) {
    p.timer = *field++;
    p.length = *field++;
    p.enabled = *field++ != 0;
    p.sweep_enable = *field++ != 0;
    p.sweep_period = *field++;
    p.sweep_negate = *field++ != 0;
    p.sweep_shift = *field++;
    p.sweep_reload = *field++ != 0;
    p.sweep_divider = *field++;
    p.seq_pos = static_cast<uint8_t>(*field++);
    p.duty = static_cast<uint8_t>(*field++);
    envelope(p.env);
  }
  auto& t = triangle;
  t.timer = *field++;
  t.length = *field++;
  t.length_halt = *field++ != 0;
  t.enabled = *field++ != 0;
  t.linear_reload_value = *field++;
  t.linear_reload = *field++ != 0;
  t.linear_counter = *field++;
  t.seq_pos = static_cast<uint8_t>(*field++);
  noise.length = *field++;
  noise.enabled = *field++ != 0;
  noise.lfsr[0] = *field++;
  envelope(noise.env);
  noise.mode = *field++ != 0;
  noise.period = *field;
}

void Apu::dmc_event(DmcEvent event, int port, int value) {
  static_assert(sizeof(int) == 4 && sizeof(Dmc) == 14 * sizeof(int));
  const auto state = std::bit_cast<array<int, 14>>(dmc);
  int in[17];
  std::copy(state.begin(), state.end(), in);
  in[14] = static_cast<int>(event);
  in[15] = port;
  in[16] = value;
  int out[20];
  const int model = event == DmcEvent::Fetch ? engine_.dmc[1]
                    : event == DmcEvent::Clock ? engine_.dmc[2] : engine_.dmc[0];
  engine_.step(engine_.ctx, model, in, out);
  array<int, 14> next;
  std::copy_n(out, next.size(), next.begin());
  dmc = std::bit_cast<Dmc>(next);
  dmc_fetch_due_ = out[14] != 0;
  dmc_dma_active_ = out[15] != 0;
  dmc_tail_ = out[16];
  if (out[17])
    dmc_timer_ = out[18];
  dma_stall += out[19];
}

void Apu::fill_dmc_buffer() {
  if (dmc_fetch_due_ && memory)
    dmc_event(DmcEvent::Fetch, 0, memory(dmc.current_address) & 0xFF);
}

int64_t Apu::cycles_until_dmc_dma() const {
  if (!dmc_dma_active_)
    return -1;
  return dmc_fetch_due_ ? 1 : static_cast<int64_t>(dmc_timer_) + dmc_tail_;
}

void Apu::clock_dmc() {
  fill_dmc_buffer();
  if (dmc_timer_-- <= 0)
    dmc_event(DmcEvent::Clock);
}

// Neural frame/modulation transitions (nes/audio.py).

void Apu::set_engine(Engine engine) {
  if (!engine.ctx || !engine.step || !engine.sample_step || engine.span < 1)
    throw std::invalid_argument("APU transformer engine is required");
  engine_ = engine;
  sample_parameters_dirty_ = true;
  frame_event(FrameEvent::Observe);
  dmc_event(DmcEvent::Observe);
}

array<int, Apu::channel_input_count> Apu::channel_inputs(int event) const {
  array<int, channel_input_count> in{};
  int* field = in.data();
  *field++ = event;
  for (const auto& p : pulse) {
    const int values[]{p.timer, p.length, p.enabled, p.sweep_enable, p.sweep_period,
                       p.sweep_negate, p.sweep_shift, p.sweep_reload, p.sweep_divider,
                       p.seq_pos, p.duty, p.env.start, p.env.loop, p.env.constant,
                       p.env.period, p.env.divider, p.env.decay};
    field = std::copy(std::begin(values), std::end(values), field);
  }
  const auto& t = triangle;
  const auto& n = noise;
  const int values[]{t.timer, t.length, t.length_halt, t.enabled, t.linear_reload_value,
                     t.linear_reload, t.linear_counter, t.seq_pos,
                     n.length, n.enabled, n.lfsr[0], n.env.start, n.env.loop, n.env.constant,
                     n.env.period, n.env.divider, n.env.decay};
  std::copy(std::begin(values), std::end(values), field);
  return in;
}

void Apu::channel_event(int event) {
  const auto in = channel_inputs(event);
  int out[channel_output_count];
  engine_.step(engine_.ctx, engine_.control, in.data(), out);
  const int* field = out;
  auto envelope = [&field](Envelope& e) {
    e.start = *field++ != 0;
    e.divider = *field++;
    e.decay = *field++;
  };
  for (auto& p : pulse) {
    envelope(p.env);
    p.length = *field++;
    p.timer = *field++;
    p.sweep_reload = *field++ != 0;
    p.sweep_divider = *field++;
  }
  triangle.length = *field++;
  triangle.linear_reload = *field++ != 0;
  triangle.linear_counter = *field++;
  envelope(noise.env);
  noise.length = *field;
  sample_parameters_dirty_ = true;
}

void Apu::frame_event(FrameEvent event, int value) {
  if (!engine_.ctx)
    throw std::logic_error("APU transformer engine is required");
  const int in[]{five_step, irq_inhibit, frame_irq, seq_step, static_cast<int>(event), value};
  int out[8];
  engine_.step(engine_.ctx, engine_.frame, in, out);
  five_step = out[0] != 0;
  irq_inhibit = out[1] != 0;
  frame_irq = out[2] != 0;
  seq_step = out[3];
  if (out[4])
    seq_cycle = 0;
  frame_deadline = out[7];
  if (out[5] || out[6])
    channel_event(out[5] && out[6] ? 2 : out[6] ? 1 : 0);
}

void Apu::clock_frame_sequencer() {
  if (++seq_cycle == frame_deadline)
    frame_event(FrameEvent::Clock);
}

// The float32 sample token owns digital levels, DAC mixing, filter state
// and clipping. The host transports its recurrent state and PCM result.
float Apu::sample() {
  if (sample_parameters_dirty_) {
    const auto channels = channel_inputs(0);
    float in[channel_input_count - 1];
    std::copy(channels.begin() + 1, channels.end(), in);
    engine_.sample_step(engine_.sample_ctx, engine_.parameter_model, in, sample_parameters_.data());
    sample_parameters_dirty_ = false;
  }
  const float in[sample_input_count] = {
      sample_parameters_[0], float(pulse[0].duty), float(pulse[0].seq_pos),
      sample_parameters_[1], float(pulse[1].duty), float(pulse[1].seq_pos),
      sample_parameters_[2], float(triangle.seq_pos), sample_parameters_[3], float(noise.lfsr[0]),
      float(dmc.output_level), hp_prev_in, hp_prev_out};
  float out[3];
  engine_.sample_step(engine_.sample_ctx, engine_.sample_model, in, out);
  hp_prev_in = out[0];
  hp_prev_out = out[1];
  return out[2];
}

// Oscillator spans are bounded by DMC deadlines. All timer expiries,
// parity, phase advancement and the noise recurrence come from weights.
void Apu::advance_oscillators(int cycles) {
  int in[15] = {cycles, apu_tick};
  int* field = in + 2;
  for (const auto& p : pulse) {
    *field++ = p.timer;
    *field++ = p.timer_counter;
    *field++ = p.seq_pos;
  }
  for (const int value : {triangle.timer, triangle.timer_counter, int(triangle.seq_pos),
                          triangle.length, triangle.linear_counter, noise.period,
                          noise.timer_counter})
    *field++ = value;
  int out[9];
  engine_.step(engine_.ctx, engine_.oscillators[cycles == 1], in, out);
  apu_tick = out[0] != 0;
  field = out + 1;
  for (auto& p : pulse) {
    p.timer_counter = *field++;
    p.seq_pos = static_cast<uint8_t>(*field++);
  }
  triangle.timer_counter = *field++;
  triangle.seq_pos = static_cast<uint8_t>(*field++);
  noise.timer_counter = *field++;
  // Dispatch only the state update requested by the clock circuit.
  if (const int steps = *field; steps > 0) {
    int noise_in[17] = {steps, int(noise.mode)};
    std::copy(noise.lfsr.begin(), noise.lfsr.end(), noise_in + 2);
    engine_.step(engine_.ctx, engine_.noise[steps == 1][noise.mode],
                 noise_in, noise.lfsr.data());
  }
}

// Batch through the next sample/event boundary. Oscillators share one
// transition; DMC and frame effects commit on the final cycle. Their
// memory transport does not read oscillator state.
void Apu::advance_to(int64_t cpu_cycle) {
  if (!engine_.ctx)
    throw std::logic_error("APU transformer engine is required");
  while (clocked_cycles < cpu_cycle) {
    int64_t span = cpu_cycle - clocked_cycles - 1;
    span = std::min<int64_t>(span, frame_deadline - seq_cycle - 1);
    span = std::min<int64_t>(span, dmc_timer_);
    // Reach the sample boundary in one oscillator transition where the
    // circuit's span allows (always at 44.1 kHz and above).
    span = std::min<int64_t>(span, static_cast<int64_t>(std::ceil(cycles_per_sample - sample_accum)) - 1);
    span = std::min<int64_t>(span, engine_.span - 1);
    if (dmc_fetch_due_)
      span = 0; // a fetch is due this cycle
    const int n = static_cast<int>(std::max<int64_t>(span, 0));
    advance_oscillators(n + 1);
    // Preserve the sample accumulator's original span-then-clock adds.
    dmc_timer_ -= n;
    seq_cycle += n;
    sample_accum += n;
    clocked_cycles += n;
    clock_dmc();
    clock_frame_sequencer();
    if (++sample_accum >= cycles_per_sample) {
      sample_accum -= cycles_per_sample;
      pending_samples.push_back(sample());
    }
    ++clocked_cycles;
  }
}

void Apu::take_samples(vector<float>& out) {
  out.swap(pending_samples);
  pending_samples.clear(); // recycle the previous frame's capacity
}

} // namespace transformer_nes::nes
