#pragma once

#include "nes/bus.h"
#include "nes/cartridge.h"
#include "nes/chr.h"
#include "nes/mappers/irq.h"
#include "nes/ppu.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace transformer_nes::nes {

struct BankControl;

using std::array;
using std::optional;
using std::pair;
using std::size_t;
using std::span;
using std::uint8_t;
using std::vector;

constexpr size_t BANK8 = 0x2000;
constexpr size_t BANK16 = 0x4000;
constexpr size_t CHR_BANK1 = 0x400;
constexpr size_t CHR_RAM_BANKS = 0x2000 / CHR_BANK1;

// Everything a mapper drives outside its own cartridge state: the PPU
// (mirroring, CHR banks, MMC5 substitution state), the machine image
// (repair pokes, bank region maps), and the shared IRQ line (acks +
// the sched_dirty re-plan signal).  The console catches the PPU up to
// each mapper access before the mapper sees it.
struct MapperHost {
  Ppu* ppu = nullptr;
  ConsoleMachine* machine = nullptr;
  IrqLine* irq = nullptr;
};

// A mapper also watches the PPU bus (PpuWatcher): scanline-IRQ mappers
// clock their counters from the PPU's A12 rises or scanline starts.
class Mapper : public PpuWatcher {
public:
  explicit Mapper(Cartridge cart) : cart_(std::move(cart)) {}
  ~Mapper() override = default;
  span<const uint8_t> prg_rom() const { return cart_.prg; }
  vector<uint8_t> take_chr() { return std::exchange(cart_.chr, {}); }
  virtual bool banked() const { return false; }
  virtual vector<uint8_t> cpu_image() const = 0;
  virtual int read(int) { throw std::runtime_error("mapper range is write-only"); }
  virtual void write(int, int) { throw std::runtime_error("mapper is not writable"); }
  virtual void write_rmw(int addr, int value, int original) {
    (void)original;
    write(addr, value);
  }
  virtual vector<pair<int, int>> register_ranges() const { return {}; }
  // Whether the PPU should report A12 rises (costs a little per fetch).
  virtual bool wants_a12() const { return false; }
  // A private neural IRQ deadline consumed over a PPU dry run;
  // null when no prediction is needed.
  virtual std::unique_ptr<IrqProbe> irq_probe() const { return nullptr; }
  virtual MapperIrq* irq_state() { return nullptr; }
  virtual BankControl* bank_state() { return nullptr; }

  void connect(MapperHost host) {
    host_ = host;
    on_connected();
  }

protected:
  Cartridge cart_;
  MapperHost host_;

  virtual void on_connected() {}

  void irq_event(MapperIrq& state, MapperIrq::Event event, int value = 0) {
    state.apply(event, value);
    if (state.irq_write)
      host_.irq->set(IRQ_MAPPER, state.level);
  }

  void repair(int addr, int value) { host_.machine->poke(addr, value & 0xFF); }
  void map_region(int base, span<const uint8_t> data) { host_.machine->map_region(base, data); }
  static span<const uint8_t> region(span<const uint8_t> data, size_t offset, size_t size) {
    return data.subspan(offset, size);
  }
};

class Nrom final : public Mapper {
public:
  using Mapper::Mapper;
  vector<uint8_t> cpu_image() const override {
    vector<uint8_t> image(0x10000);
    std::copy(cart_.prg.begin(), cart_.prg.end(), image.begin() + 0x8000);
    if (cart_.prg.size() == BANK16)
      std::copy(cart_.prg.begin(), cart_.prg.end(), image.begin() + 0xC000);
    return image;
  }
};

} // namespace transformer_nes::nes
