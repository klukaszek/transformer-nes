#pragma once

#include "nes/ppu.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace transformer_nes::nes {

// mappers.py's caller-owned state and token ABI. The circuit also emits
// its next effect in qualified-edge space; the PPU supplies absolute dots.
struct MapperIrq {
  enum Chip { Mmc3, Mmc5 };
  enum Event { Clock, Idle, Target, Reload, Enable, Disable, Ack, event_count };
  struct Engine {
    void* ctx = nullptr;
    void (*step)(void*, int, const int*, int*) = nullptr;
    int model = 0;
  } engine;
  explicit MapperIrq(Chip chip_in) : chip(chip_in) {}
  Chip chip;
  // counter, target, phase (reload / in-frame), pending, enabled.
  std::array<int, 5> state{};
  int level = 0, read_value = 0, predictable = 0, next_edge = 0, irq_write = 0;

  void apply(Event event, int value = 0) {
    if (!engine.step)
      throw std::runtime_error("mapper IRQ transformer is required");
    const int in[]{state[0], state[1], state[2], state[3], state[4], event, value};
    int out[10];
    engine.step(engine.ctx, engine.model, in, out);
    std::copy_n(out, state.size(), state.begin());
    level = out[5];
    read_value = out[6];
    predictable = out[7];
    next_edge = out[8];
    irq_write = out[9];
  }
};

// A preview counts qualified bus events against a neural effect deadline.
// This is a scheduling budget, not a second implementation of the counter.
// An MMC5 idle event starts a new frame epoch, so ask the same circuit for
// its new deadline on private state. Preview never commits mapper state.
class MapperIrqProbe final : public IrqProbe {
public:
  explicit MapperIrqProbe(const MapperIrq& state) : state_(state), remaining_(state.next_edge) {}
  void ppu_event(const PpuEvent& event) override {
    const auto clock = state_.chip == MapperIrq::Mmc3 ? PpuEvent::A12Rise : PpuEvent::ScanlineStart;
    if (event.kind == clock) {
      if (remaining_ > 0 && --remaining_ == 0 && fire_dot < 0)
        fire_dot = event.dot;
    } else if (state_.chip == MapperIrq::Mmc5 && event.kind == PpuEvent::Idle) {
      state_.apply(MapperIrq::Idle);
      remaining_ = state_.next_edge;
    }
  }

private:
  MapperIrq state_;
  int remaining_;
};

} // namespace transformer_nes::nes
