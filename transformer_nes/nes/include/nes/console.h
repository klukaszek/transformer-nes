#pragma once

#include "nes/apu.h"
#include "nes/bus.h"
#include "nes/cartridge.h"
#include "nes/mappers/mappers.h"
#include "nes/ppu.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace transformer_nes::nes {

using std::int64_t;
using std::size_t;
using std::uint8_t;
using std::unique_ptr;
using std::vector;

// NTSC frame timing in CPU cycles at 1.789773 MHz.
constexpr int64_t vblank_start_cycles = 27'394;
constexpr int64_t frame_cycles = 29'781;
constexpr int64_t scanline_cycle(int line) { return static_cast<int64_t>(line + 1) * 341 / 3; }

// Visible-span scanline containing a console-clock cycle, -1 outside.
constexpr int span_scanline(int64_t cycle, int64_t frame_base) {
  const int64_t rel = cycle - frame_base;
  if (rel < 0 || rel >= vblank_start_cycles)
    return -1;
  const int row = static_cast<int>(rel * 3 / 341);
  return row < 240 ? row : -1;
}

class Console {
public:
  Console(unique_ptr<Mapper> mapper, Ppu ppu, Apu apu, ConsoleMachine& machine);
  Ppu::Frame run_frame();
  Ppu::Frame run(int frames);
  void set_button(size_t index, bool pressed) { controller.set_button(index, pressed); }

  unique_ptr<Mapper> mapper;
  Ppu ppu;
  Apu apu;
  ConsoleMachine& machine;
  Controller controller;
  vector<float> last_audio;
  int64_t cycles = 0;
  int frame_index = 0;
  bool halted = false;

private:
  // Device-table adapters: route MMIO ranges to the owning chip and
  // convert the write's cycle stamp to a visible-span scanline (the
  // frame_base reference tracks the console's live frame origin).
  class PpuDevice final : public Device {
  public:
    PpuDevice(Ppu& ppu, const int64_t& frame_base) : ppu_(ppu), frame_base_(frame_base) {}
    int read(int address) override { return ppu_.read(address); }
    void write(int address, int value, int64_t cycle) override {
      ppu_.write(address, value, span_scanline(cycle, frame_base_));
    }
    ReadLease read_lease(int a) override {
      auto v = ppu_.read_lease(a);
      return v ? ReadLease(*v) : ReadLease{};
    }

  private:
    Ppu& ppu_;
    const int64_t& frame_base_;
  };
  class ApuDevice final : public Device {
  public:
    explicit ApuDevice(Apu& apu) : apu_(apu) {}
    int read(int address) override { return apu_.read(address); }
    void write(int address, int value, int64_t) override { apu_.write(address, value); }

  private:
    Apu& apu_;
  };
  class MapperDevice final : public Device {
  public:
    MapperDevice(Mapper& m, const int64_t& frame_base) : m_(m), frame_base_(frame_base) {}
    int read(int a) override { return m_.read(a); }
    void write(int a, int v, int64_t cycle) override {
      m_.write(a, v, span_scanline(cycle, frame_base_));
    }
    void write_rmw(int a, int original, int value) override { m_.write_rmw(a, original, value); }

  private:
    Mapper& m_;
    const int64_t& frame_base_;
  };
  // $4017 read is the APU frame counter (open-bus 0x40 leases);
  // writes program the frame sequencer.
  class FrameCounter final : public Device {
  public:
    explicit FrameCounter(Apu& apu) : apu_(apu) {}
    int read(int address) override { return apu_.read(address); }
    void write(int address, int value, int64_t) override { apu_.write(address, value); }
    ReadLease read_lease(int) override { return 0x40; }

  private:
    Apu& apu_;
  };

  void attach_devices();
  void run_until(int64_t target, bool replan = false);
  void check_interrupts();

  IrqLine irq_;
  int64_t frame_base_ = 0;
  int sprite0_scanline_ = -1;
  OamDma dma_;
  PpuDevice ppu_device_;
  ApuDevice apu_device_;
  MapperDevice mapper_device_;
  FrameCounter frame_counter_;
};

} // namespace transformer_nes::nes
