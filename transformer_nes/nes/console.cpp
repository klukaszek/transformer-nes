#include "nes/console.h"

#include <algorithm>
#include <utility>

namespace transformer_nes::nes {

Console::Console(unique_ptr<Mapper> mapper_in, Ppu ppu_in, Apu apu_in, ConsoleMachine& machine_in)
    : mapper(std::move(mapper_in)), ppu(std::move(ppu_in)), apu(std::move(apu_in)),
      machine(machine_in), dma_(machine, ppu, irq_, ppu_device_), ppu_device_(ppu, irq_, horizon_),
      apu_device_(apu), mapper_device_(*mapper, ppu, machine, ppu_device_) {
  irq_.machine = &machine;
  apu.memory = [&machine = machine](int address) { return machine.read_ram(address); };
  attach_devices();
}

void Console::attach_devices() {
  machine.attach(ppu_device_, 0x2000, 0x3FFF);
  machine.attach(dma_, 0x4014, 0x4014);
  machine.attach(apu_device_, 0x4000, 0x4013);
  machine.attach(apu_device_, 0x4015, 0x4015);
  machine.attach(controller, 0x4016, 0x4016);
  machine.attach(apu_device_, 0x4017, 0x4017);
  ppu.set_watcher(mapper.get(), mapper->wants_a12());

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
// Runs the CPU to `target` cycles in chunks that end at each DMC sample
// fetch (its stall lands there).  A chunk is cut short when a DMC
// register write moves the fetch schedule, an NMI is pending, or a PPU
// or mapper write may have moved the mapper-IRQ prediction; for that
// last one the run stops and the frame loop re-plans.
void Console::run_until(int64_t target) {
  if (halted || target <= cycles)
    return;
  bool dirty = false;
  while (cycles < target && !halted) {
    const int64_t dmc_until = apu.cycles_until_dmc_dma();
    const int64_t dmc_target = dmc_until >= 0 ? cycles + dmc_until : target;
    const int64_t chunk_target = std::min(target, dmc_target);
    auto out = machine.run_cycles(
        std::max<int64_t>(1, chunk_target - cycles),
        [this, &dirty] {
          check_interrupts();
          dirty = std::exchange(irq_.sched_dirty, false);
          return dirty || ppu.delayed_nmi_pending() ||
                 std::exchange(apu_device_.dmc_moved, false);
        },
        cycles);
    if (out.cycles <= 0)
      break;
    cycles += out.cycles;
    halted = out.halted;
    apu.advance_to(cycles);
    if (const int stall = apu.take_dma_stall()) {
      cycles += stall;
      apu.advance_to(cycles);
    }
    if (ppu.take_delayed_nmi()) {
      // Run exactly one more instruction before the NMI.
      const auto next = machine.run_cycles(1, {}, cycles);
      cycles += next.cycles;
      halted = next.halted;
      machine.set_nmi();
    }
    if (dirty)
      break;
  }
  check_interrupts();
}

// One frame: from line 0 through the pre-render line.  Each iteration
// catches the PPU up to the CPU, then runs the CPU to the next event
// that must interrupt it -- vblank (NMI), the frame's end, or the
// mapper IRQ a PPU dry run predicts.  Sprite-0 hits and every other
// PPU-visible effect are resolved lazily at the access that sees them.
Ppu::Frame Console::run_frame() {
  const int64_t start = ppu.frame_start();
  while (!halted) {
    ppu.catch_up(cycles * 3);
    check_interrupts();
    if (ppu.frame_start() != start)
      break;
    int64_t event = ppu.next_frame_start();
    if (ppu.position() <= ppu.vblank_dot())
      event = std::min(event, ppu.vblank_dot());
    if (auto probe = mapper->irq_probe())
      if (const int64_t fire = ppu.preview(event, probe.get(), false); fire >= 0)
        event = fire;
    horizon_ = event;
    // The CPU samples NMI/IRQ before an instruction's last cycle, and
    // the PPU's NMI output lags the flag by a dot: stop on the first
    // boundary whose last cycle starts after that (calibrated against
    // blargg's ppu_vbl_nmi 05-nmi_timing).
    run_until((event + 1) / 3 + 2);
    machine.clear_read_lease();
  }
  ++frame_index;
  last_audio = apu.take_samples();
  irq_.set(IRQ_APU, apu.irq_pending());
  return ppu.completed_frame();
}

} // namespace transformer_nes::nes
