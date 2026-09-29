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

// The PPU clock runs at 3 dots per CPU cycle.  An access by an
// instruction ending at console cycle `cycle` lands mid-way through its
// last cycle.
constexpr int64_t access_dot(int64_t cycle) { return cycle * 3 - 2; }

class Console {
public:
  Console(unique_ptr<Mapper> mapper, Ppu ppu, Apu apu, ConsoleMachine& machine);
  Ppu::Frame run_frame();
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
  // Device-table adapters: route MMIO ranges to the owning chip.  PPU
  // and mapper accesses carry their access dot so the PPU catches up
  // first; writes that may move the mapper-IRQ prediction ask the
  // console to re-plan.
  class PpuDevice final : public Device {
  public:
    PpuDevice(Ppu& ppu, IrqLine& irq, const int64_t& horizon)
        : ppu_(ppu), irq_(irq), horizon_(horizon) {}
    int read(int address, int64_t cycle) override { return ppu_.read(address, access_dot(cycle)); }
    void write(int address, int value, int64_t cycle) override {
      ppu_.write(address, value, access_dot(cycle));
      if (ppu_.take_schedule_change())
        irq_.sched_dirty = true;
    }
    ReadLease read_lease(int a) override {
      auto v = ppu_.read_lease(a, horizon_);
      return v ? ReadLease(*v) : ReadLease{};
    }
    // Stop serving the lease 7 cycles (one instruction) before a read
    // could observe the change.
    int64_t lease_expiry() const override {
      const int64_t dot = ppu_.lease_expiry();
      return dot == NEVER ? NEVER : (dot + 4) / 3 - 7;
    }

  private:
    Ppu& ppu_;
    IrqLine& irq_;
    const int64_t& horizon_;
  };
  // $4000-$4015 and the frame counter at $4017, whose reads lease the
  // open-bus 0x40 (controller 2 is not connected).
  class ApuDevice final : public Device {
  public:
    explicit ApuDevice(Apu& apu) : apu_(apu) {}
    int read(int address, int64_t) override { return apu_.read(address); }
    void write(int address, int value, int64_t cycle) override {
      if (cycle >= 0)
        apu_.advance_to(cycle);
      apu_.write(address, value);
      // A DMC register can start, stop or retime its sample fetches: the
      // CPU run is cut so the next chunk ends at the new fetch (and each
      // fetch's stall lands when it happens, not lumped at the chunk's end).
      if ((address >= 0x4010 && address <= 0x4013) || address == 0x4015)
        dmc_moved = true;
    }
    ReadLease read_lease(int address) override {
      return address == 0x4017 ? ReadLease(0x40) : ReadLease{};
    }
    bool dmc_moved = false;

  private:
    Apu& apu_;
  };
  // A mapper write can change CHR banks under a pending sprite-0 hit,
  // so it also withdraws the PPU's $2002 lease.
  class MapperDevice final : public Device {
  public:
    MapperDevice(Mapper& m, Ppu& ppu, ConsoleMachine& machine, Device& ppu_device)
        : m_(m), ppu_(ppu), machine_(machine), ppu_device_(ppu_device) {}
    int read(int a, int64_t cycle) override {
      ppu_.catch_up(access_dot(cycle));
      return m_.read(a);
    }
    void write(int a, int v, int64_t cycle) override {
      ppu_.catch_up(access_dot(cycle));
      machine_.drop_leases(ppu_device_);
      m_.write(a, v);
    }
    void write_rmw(int a, int value, int64_t cycle) override {
      ppu_.catch_up(access_dot(cycle));
      machine_.drop_leases(ppu_device_);
      m_.write_rmw(a, value);
    }

  private:
    Mapper& m_;
    Ppu& ppu_;
    ConsoleMachine& machine_;
    Device& ppu_device_;
  };
  // $4014: copy a 256-byte CPU page into OAM, stalling for 513 or 514 cycles.
  class OamDma final : public Device {
  public:
    OamDma(ConsoleMachine& machine, Ppu& ppu, IrqLine& irq, Device& ppu_device)
        : machine_(machine), ppu_(ppu), irq_(irq), ppu_device_(ppu_device) {}
    int read(int, int64_t) override { return 0; }
    void write(int, int value, int64_t cycle) override {
      array<uint8_t, 256> page{};
      const int base = (value & 0xFF) << 8;
      for (size_t i = 0; i < page.size(); ++i)
        page[i] = static_cast<uint8_t>(machine_.read_ram(base + static_cast<int>(i)));
      ppu_.oam_dma(page, access_dot(cycle));
      machine_.drop_leases(ppu_device_); // sprite 0 may have moved
      if (ppu_.take_schedule_change())
        irq_.sched_dirty = true;
      stall_ += 513 + static_cast<int>(cycle & 1);
    }
    int take_stall() override {
      const int out = stall_;
      stall_ = 0;
      return out;
    }

  private:
    ConsoleMachine& machine_;
    Ppu& ppu_;
    IrqLine& irq_;
    Device& ppu_device_;
    int stall_ = 0;
  };

  void attach_devices();
  void run_until(int64_t target);
  void check_interrupts();

  IrqLine irq_;
  // The dot the current CPU run stops at: $2002 leases need not look
  // past it.
  int64_t horizon_ = 0;
  OamDma dma_;
  PpuDevice ppu_device_;
  ApuDevice apu_device_;
  MapperDevice mapper_device_;
};

} // namespace transformer_nes::nes
