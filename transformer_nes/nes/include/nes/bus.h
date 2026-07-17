#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
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

struct CpuWrite {
  int addr = 0;
  int value = 0;
};

struct CpuStep {
  int pc = 0, a = 0, x = 0, y = 0, s = 0, p = 0;
  bool halted = false;
  vector<CpuWrite> writes;
};

class SteppedRuntime {
public:
  virtual ~SteppedRuntime() = default;
  virtual int pc() const = 0;
  virtual int x() const = 0;
  virtual int y() const = 0;
  virtual bool halted() const = 0;
  virtual bool nmi_pending() const = 0;
  virtual bool irq_ready() const = 0;
  virtual void set_nmi() = 0;
  virtual void set_irq() = 0;
  virtual void clear_irq() = 0;
  virtual void poke(int addr, int value) = 0;
  virtual CpuStep step() = 0;
};

using ReadLease = variant<monostate, int, vector<uint8_t>>;
class Device {
public:
  virtual ~Device() = default;
  virtual int read(int addr) = 0;
  // cycle: the console-clock stamp the write executed at (known because
  // watched writes pause the engine, -1 when no stamp exists); devices
  // that model mid-frame state (the PPU) use it, the rest ignore it.
  virtual void write(int addr, int value, int64_t cycle) = 0;
  virtual void write_rmw(int addr, int, int value) { write(addr, value, -1); }
  virtual int take_stall() { return 0; }
  virtual ReadLease read_lease(int) { return {}; }
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
  // writes carry an absolute cycle stamp (write_at).  A true return
  // from the pause callback stops the run early (the console re-plans
  // its frame events after a mid-frame IRQ-schedule change).
  virtual MachineRun run_cycles(int64_t, function<bool()> = {}, int64_t cycle_base = 0) = 0;
  virtual int read_ram(int) const = 0;
  virtual void poke(int, int) = 0;
  virtual void map_region(int, span<const uint8_t>) = 0;
};

struct OpcodeInfo {
  int cycles = 0;
  int read_mode = -1;
};
using OpcodeTable = array<OpcodeInfo, 256>;

class Bus : public ConsoleMachine {
public:
  Bus(SteppedRuntime& runtime, span<const uint8_t> image, const OpcodeTable& opcodes);
  void attach(Device& device, int lo, int hi, bool writes_only = false) override;
  void set_nmi() override { runtime_.set_nmi(); }
  void set_irq() override { runtime_.set_irq(); }
  void clear_irq() override { runtime_.clear_irq(); }
  bool halted() const { return runtime_.halted(); }
  CpuStep step();
  vector<CpuStep> run(int max_steps, const vector<int>& nmi_at = {},
                      const vector<int>& irq_at = {});
  void clear_read_lease() override {}
  MachineRun run_cycles(int64_t budget, function<bool()> pause = {},
                        int64_t cycle_base = 0) override;
  int read_ram(int addr) const override { return ram_[address_index(addr)]; }
  void poke(int addr, int value) override {
    runtime_.poke(addr, value);
    poke_ram(addr, value);
  }
  void map_region(int base, span<const uint8_t> data) override;
  void poke_ram(int addr, int value) { ram_[address_index(addr)] = static_cast<uint8_t>(value); }
  const array<uint8_t, 0x10000>& ram() const { return ram_; }
  int64_t cycles() const { return cycles_; }

private:
  static constexpr size_t address_index(int address) { return static_cast<uint16_t>(address); }
  optional<int> read_target(int pc, int x, int y) const;
  SteppedRuntime& runtime_;
  OpcodeTable opcodes_{};
  array<uint8_t, 0x10000> ram_{};
  DeviceMap devices_;
  int64_t cycles_ = 0;
  int64_t stamp_base_ = 0; // console-clock minus cycles_, set per run_cycles
};

class Controller final : public Device {
public:
  void set_button(size_t index, bool pressed);
  int read(int addr) override;
  void write(int addr, int value, int64_t) override;
  ReadLease read_lease(int addr) override;
  void lease_consumed(int addr, int count) override;

private:
  array<bool, 8> buttons_{};
  bool strobe_ = false;
  size_t index_ = 0;
};

class OamDma final : public Device {
public:
  OamDma(function<int(int)> read_cpu, function<void(span<const uint8_t, 256>)> write_oam)
      : read_cpu_(std::move(read_cpu)), write_oam_(std::move(write_oam)) {}
  int read(int) override { return 0; }
  void write(int, int value, int64_t) override;
  int take_stall() override {
    int out = stall_;
    stall_ = 0;
    return out;
  }

private:
  function<int(int)> read_cpu_;
  function<void(span<const uint8_t, 256>)> write_oam_;
  int stall_ = 0;
};

} // namespace transformer_nes::nes
