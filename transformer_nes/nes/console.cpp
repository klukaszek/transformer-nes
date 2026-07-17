#include "nes/console.h"

#include <algorithm>
#include <utility>

namespace transformer_nes::nes {

namespace {

constexpr int max_frame_events = 32;

} // namespace

Console::Console(unique_ptr<Mapper> mapper_in, Ppu ppu_in, Apu apu_in, ConsoleMachine& machine_in)
    : mapper(std::move(mapper_in)), ppu(std::move(ppu_in)), apu(std::move(apu_in)),
      machine(machine_in), dma_(machine, ppu), ppu_device_(ppu, frame_base_), apu_device_(apu),
      mapper_device_(*mapper, frame_base_), frame_counter_(apu) {
  irq_.machine = &machine;
  attach_devices();
}

void Console::attach_devices() {
  machine.attach(ppu_device_, 0x2000, 0x3FFF);
  machine.attach(dma_, 0x4014, 0x4014);
  machine.attach(apu_device_, 0x4000, 0x4013);
  machine.attach(apu_device_, 0x4015, 0x4015);
  machine.attach(controller, 0x4016, 0x4016);
  machine.attach(frame_counter_, 0x4017, 0x4017);

  if (!mapper->banked())
    return;
  for (auto [lo, hi] : mapper->register_ranges())
    machine.attach(mapper_device_, lo, hi);
  machine.attach(mapper_device_, 0x8000, 0xFFFF, true);
  mapper->connect({&ppu, &machine, &irq_});
}

void Console::check_interrupts() {
  if (ppu.take_nmi_edge())
    machine.set_nmi();
  if ((irq_.sources & IRQ_APU) != 0 && !apu.irq_pending())
    irq_.set(IRQ_APU, false);
}
// A replan run stops early (cycles short of target) when a mapper
// write moved the IRQ schedule: the event-planning loop re-evaluates.
// Plain runs always complete -- cutting the vblank run short would
// slip the NMI's tail into the next frame's visible span.
void Console::run_until(int64_t target, bool replan) {
  if (halted || target <= cycles)
    return;
  auto out = machine.run_cycles(
      target - cycles,
      [this, replan] {
        check_interrupts();
        const bool dirty = std::exchange(irq_.sched_dirty, false);
        return replan && dirty;
      },
      cycles);
  cycles += out.cycles;
  halted = out.halted;
}
Ppu::Frame Console::run_frame() {
  const int64_t base = frame_base_;
  const int64_t span_end = base + vblank_start_cycles;
  mapper->begin_frame();
  bool fired = false;
  int last_irq = -1;
  for (int guard = 0; guard < max_frame_events && cycles < span_end && !halted; ++guard) {
    // Next event: the sprite-0 hit or the mapper IRQ, earliest first
    // (sprite-0 wins a tie -- the SMB-style poll precedes the handler).
    int line = -1, kind = -1;
    if (!fired && sprite0_scanline_ >= 0 && ppu.rendering_enabled()) {
      line = sprite0_scanline_;
      kind = 0;
    }
    auto irq = mapper->irq_scanline(last_irq);
    if (irq && *irq > last_irq && ppu.rendering_enabled() && (kind < 0 || *irq < line)) {
      line = *irq;
      kind = 1;
    }
    if (kind < 0) {
      // No event pending; a mid-frame arm stops this run early and
      // re-enters the planning loop.
      run_until(span_end, true);
      continue;
    }
    const int64_t at = std::min(base + scanline_cycle(line), span_end);
    run_until(at, true);
    if (cycles < at)
      continue; // IRQ schedule moved mid-run; re-plan
    machine.clear_read_lease();
    if (!kind) {
      fired = true;
      ppu.sprite0_hit = true;
      // Only post-hit scroll writes can be the game's split batch; their
      // rows arrive exactly via the stamped write path.
      ppu.begin_span_watch();
      continue;
    }
    // Re-validate with the pre-event context: the delta chain's next
    // line is still computed from the PREVIOUS expiry.
    auto current = mapper->irq_scanline(last_irq);
    last_irq = line;
    if (ppu.rendering_enabled() && current && *current == line) {
      ppu.mark_split(line);
      mapper->notify_scanline_irq();
      irq_.set(IRQ_MAPPER, true);
      run_until(std::min(at + scanline_cycle(1), span_end));
    }
  }
  run_until(span_end);
  machine.clear_read_lease();
  auto frame = ppu.render_frame();
  sprite0_scanline_ = ppu.sprite0_scanline;
  ppu.begin_vblank();
  if (ppu.nmi_enabled())
    machine.set_nmi();
  run_until(base + frame_cycles);
  machine.clear_read_lease();
  ppu.end_vblank();
  frame_base_ += frame_cycles;
  ++frame_index;
  last_audio = apu.end_frame(frame_cycles, [this](int address) { return machine.read_ram(address); });
  cycles += apu.take_dma_stall();
  irq_.set(IRQ_APU, apu.irq_pending());
  return frame;
}
Ppu::Frame Console::run(int frames) {
  Ppu::Frame frame{};
  for (int i = 0; i < frames && !halted; ++i)
    frame = run_frame();
  return frame;
}

} // namespace transformer_nes::nes
