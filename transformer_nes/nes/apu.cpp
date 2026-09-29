// Native NES APU (2A03): MMIO registers, channel state, frame sequencer,
// sample generation, and DMC memory reads on the CPU-clock timeline.

#include <algorithm>
#include <utility>

#include "nes/apu.h"

namespace transformer_nes::nes {

// Device protocol ($4000-$4017)

int Apu::read(int addr) {
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

void Apu::write(int addr, int value) {
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

void Apu::fill_dmc_buffer() {
  if (!dmc.buffer_empty || dmc.bytes_remaining == 0 || !memory)
    return;
  dmc.sample_buffer = memory(dmc.current_address) & 0xFF;
  dmc.buffer_empty = false;
  dma_stall += 4;
  dmc.current_address = dmc.current_address == 0xFFFF ? 0x8000 : uint16_t(dmc.current_address + 1);
  if (--dmc.bytes_remaining == 0) {
    if (dmc.loop)
      dmc.restart();
    else if (dmc.irq_enabled)
      dmc.irq = true;
  }
}

int64_t Apu::cycles_until_dmc_dma() const {
  if (dmc.bytes_remaining == 0)
    return -1;
  if (dmc.buffer_empty)
    return 1;
  return static_cast<int64_t>(dmc.timer_counter) + 2 +
         static_cast<int64_t>(dmc.bits_remaining - 1) * dmc.period;
}

void Apu::clock_dmc() {
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

// Frame sequencer

void Apu::clock_quarter() { // envelopes + triangle linear counter
  pulse[0].env.clock();
  pulse[1].env.clock();
  noise.env.clock();
  triangle.clock_linear();
}
void Apu::clock_half() { // length counters + sweeps
  pulse[0].clock_length();
  pulse[1].clock_length();
  triangle.clock_length();
  noise.clock_length();
  pulse[0].clock_sweep();
  pulse[1].clock_sweep();
}

// Advance the sequencer by one CPU cycle, firing its steps at the NTSC
// trigger cycles (expressed in CPU cycles = 2x the APU-cycle values).
void Apu::clock_frame_sequencer() {
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

// Mixing (2A03 nonlinear DAC + DC blocker)

// Raw mixed level in ~[0, 1] from the two nonlinear DAC groups.
double Apu::mix_level() const {
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
float Apu::sample() {
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
void Apu::clock_cycle() {
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

// Runs to `cpu_cycle` in spans with nothing but counting in them: each
// ends before the next sample, frame-sequencer step, DMC timer expiry or
// DMC fetch, and those cycles run one at a time through clock_cycle.  A
// span's timers advance arithmetically, exactly as cycle-by-cycle.
void Apu::advance_to(int64_t cpu_cycle) {
  while (clocked_cycles < cpu_cycle) {
    const auto& events = five_step ? five_step_cycles : four_step_cycles;
    int64_t span = cpu_cycle - clocked_cycles;
    span = std::min<int64_t>(span, events[seq_step] - seq_cycle - 1);
    span = std::min<int64_t>(span, dmc.timer_counter);
    // The cycle whose count reaches a sample runs singly.
    span = std::min<int64_t>(span, static_cast<int64_t>(cycles_per_sample - sample_accum - 1.0));
    if (dmc.buffer_empty && dmc.bytes_remaining > 0)
      span = 0; // a fetch is due this cycle
    if (span <= 0) {
      clock_cycle();
      if (++sample_accum >= cycles_per_sample) {
        sample_accum -= cycles_per_sample;
        pending_samples.push_back(sample());
      }
      ++clocked_cycles;
      continue;
    }
    const int n = static_cast<int>(span);
    triangle.advance(n);
    dmc.timer_counter -= n;
    const int apu_ticks = apu_tick ? n / 2 : (n + 1) / 2;
    apu_tick = apu_tick != ((n & 1) != 0);
    pulse[0].advance(apu_ticks);
    pulse[1].advance(apu_ticks);
    noise.advance(apu_ticks);
    seq_cycle += n;
    sample_accum += n;
    clocked_cycles += n;
  }
}

vector<float> Apu::take_samples() {
  vector<float> out;
  out.swap(pending_samples);
  return out;
}


} // namespace transformer_nes::nes
