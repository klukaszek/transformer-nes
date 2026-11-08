#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::function;
using std::int64_t;
using std::monostate;
using std::optional;
using std::size_t;
using std::span;
using std::uint16_t;
using std::uint8_t;
using std::variant;
using std::vector;

constexpr size_t cpu_ram_index(int address) {
  const auto a = static_cast<uint16_t>(address);
  return a < 0x2000 ? a & 0x07FF : a;
}

struct CpuWrite {
  int addr = 0;
  int value = 0;
  int original = -1; // RMW operand supplied by the CPU; -1 for ordinary writes
};

class SteppedRuntime {
public:
  virtual ~SteppedRuntime() = default;
  virtual bool halted() const = 0;
  virtual void set_nmi() = 0;
  virtual void set_irq() = 0;
  virtual void clear_irq() = 0;
  virtual void poke(int addr, int value) = 0;
  // The next step ahead of running it: its cycles, and the device-visible
  // memory read it issues (-1: none), which the bus serves first.
  struct Next {
    int cycles = 0;
    int read = -1;
  };
  virtual Next peek() const = 0;
  virtual vector<CpuWrite> step() = 0;
};

struct ReadReply {
  array<uint8_t, 8> bytes{};
  int size = 0;
};
using ReadLease = variant<monostate, int, ReadReply>;
class Device {
public:
  virtual ~Device() = default;
  // cycle: the console-clock cycle the accessing instruction ends at
  // (known because watched accesses pause the engine); devices that
  // model time (the PPU, mappers watching it) use it, the rest ignore it.
  virtual int read(int addr, int64_t cycle) = 0;
  virtual void write(int addr, int value, int64_t cycle) = 0;
  virtual void write_rmw(int addr, int value, int original, int64_t cycle) {
    (void)original;
    write(addr, value, cycle);
  }
  virtual int take_stall() { return 0; }
  virtual ReadLease read_lease(int) { return {}; }
  // The console cycle from which the lease just granted may be stale;
  // the machine stops serving it before then.
  virtual int64_t lease_expiry() const { return std::numeric_limits<int64_t>::max(); }
  virtual void lease_consumed(int, int) {}
};

struct MachineRun {
  int64_t cycles = 0;
  bool halted = false;
};

class DeviceMap final {
public:
  void attach(Device& device, int first, int last, bool writes_only = false);
  [[nodiscard]] Device* find(int address, bool for_read = false) const;

private:
  struct Mapping {
    int first = 0;
    int last = 0;
    Device* device = nullptr;
    bool writes_only = false;
  };

  vector<Mapping> mappings_;
};

class ConsoleMachine {
public:
  virtual ~ConsoleMachine() = default;
  virtual void attach(Device&, int, int, bool = false) = 0;
  virtual void set_nmi() = 0;
  virtual void set_irq() = 0;
  virtual void clear_irq() = 0;
  virtual void clear_read_lease() = 0;
  // cycle_base: the console-clock value at entry, so drained device
  // writes carry an absolute cycle stamp.  `pause` is polled where the
  // run stops between instructions (after every one on the stepped Bus,
  // after serviced writes on the engine); true ends the run there.
  virtual MachineRun run_cycles(int64_t, function<bool()> = {}, int64_t cycle_base = 0) = 0;
  virtual int read_ram(int) const = 0;
  virtual void poke(int, int) = 0;
  virtual void map_region(int, span<const uint8_t>) = 0;
  // Withdraw any read lease a device granted (its value may be stale).
  virtual void drop_leases(Device&) {}
};

// The shared level-sensitive IRQ line: sources OR onto the machine's
// one line, so acking one source cannot clear another's pending level.
// sched_dirty is the mapper's "my IRQ schedule moved" signal; the
// console's frame-event planning loop consumes it.
constexpr uint8_t IRQ_MAPPER = 1;
constexpr uint8_t IRQ_APU = 2;
struct IrqLine {
  ConsoleMachine* machine = nullptr;
  uint8_t sources = 0;
  bool sched_dirty = false;

  void set(uint8_t bit, bool active) {
    if (active)
      sources |= bit;
    else
      sources &= static_cast<uint8_t>(~bit);
    if (sources == 0)
      machine->clear_irq();
    else
      machine->set_irq();
  }
};

class Bus : public ConsoleMachine {
public:
  Bus(SteppedRuntime& runtime, span<const uint8_t> image);
  void attach(Device& device, int lo, int hi, bool writes_only = false) override;
  void set_nmi() override { runtime_.set_nmi(); }
  void set_irq() override { runtime_.set_irq(); }
  void clear_irq() override { runtime_.clear_irq(); }
  bool halted() const { return runtime_.halted(); }
  void step();
  void clear_read_lease() override {}
  MachineRun run_cycles(int64_t budget, function<bool()> pause = {},
                        int64_t cycle_base = 0) override;
  int read_ram(int addr) const override { return ram_[cpu_ram_index(addr)]; }
  void poke(int addr, int value) override {
    runtime_.poke(addr, value);
    poke_ram(addr, value);
  }
  void map_region(int base, span<const uint8_t> data) override;
  void poke_ram(int addr, int value) { ram_[cpu_ram_index(addr)] = static_cast<uint8_t>(value); }
  int64_t cycles() const { return cycles_; }

private:
  SteppedRuntime& runtime_;
  array<uint8_t, 0x10000> ram_{};
  DeviceMap devices_;
  int64_t cycles_ = 0;
  int64_t stamp_base_ = 0; // console-clock minus cycles_, set per run_cycles
};

class Controller final : public Device {
public:
  enum Event { Write, Read, Lease, Consume, EventCount };
  struct Engine {
    void* ctx = nullptr;
    void (*step)(void*, int, const int*, int*) = nullptr;
    int model = 0;
  } engine;
  void set_button(size_t index, bool pressed);
  int read(int addr, int64_t) override;
  void write(int addr, int value, int64_t) override;
  ReadLease read_lease(int addr) override;
  void lease_consumed(int addr, int count) override;
private:
  array<int, 8> buttons_{};
  array<int, 2> state_{};
  array<int, 12> apply(Event event, int value = 0, int count = 0);
};

} // namespace transformer_nes::nes
